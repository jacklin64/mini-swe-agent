import os
import platform
import signal
import subprocess
import threading
import time
from typing import Any

from pydantic import BaseModel

from minisweagent.exceptions import Submitted
from minisweagent.utils.serialize import recursive_merge


class LocalEnvironmentConfig(BaseModel):
    cwd: str = ""
    env: dict[str, str] = {}
    timeout: int = 30
    memory_limit_gb: float | None = None
    """Per-command memory cap (GiB, summed over every process the command started, including
    background ones that outlive it). A command over the cap has its whole process group
    killed and the agent is told why. None disables it (upstream behaviour)."""
    memory_poll_interval: float = 0.5
    """Seconds between memory checks when memory_limit_gb is set."""
    kill_process_group_on_timeout: bool = False
    """On timeout, kill everything the command started, not just the /bin/sh wrapper.
    Without it, e.g. `pytest ... | tail` leaves pytest running after the timeout."""


_PAGE_SIZE = os.sysconf("SC_PAGE_SIZE")
_GIB = 1024**3


def _session_processes() -> dict[int, list[int]]:
    """Map session id -> pids, from /proc (Linux only)."""
    sessions: dict[int, list[int]] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat", "rb") as f:
                stat = f.read()
        except OSError:
            continue
        # comm (field 2) may contain spaces/parens; the fields after it start at the last ')'.
        fields = stat[stat.rfind(b")") + 2 :].split()
        sessions.setdefault(int(fields[3]), []).append(int(entry))
    return sessions


def _process_memory(pid: int) -> int:
    """Proportional set size in bytes (shared pages split between sharers), RSS fallback."""
    try:
        with open(f"/proc/{pid}/smaps_rollup", "rb") as f:
            for line in f:
                if line.startswith(b"Pss:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    try:
        with open(f"/proc/{pid}/statm", "rb") as f:
            return int(f.read().split()[1]) * _PAGE_SIZE
    except OSError:
        return 0


def _process_cmdline(pid: int) -> str:
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().replace(b"\0", b" ").decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def _killpg(sid: int):
    try:
        os.killpg(sid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class _MemoryWatchdog:
    """Polls the memory of every session the environment started and kills any over the cap.

    Runs in a daemon thread so background processes are policed too, including while the
    model is thinking between commands (a runaway grew ~0.6 GB/s in practice).
    """

    def __init__(self, limit_bytes: int, interval: float):
        self.limit_bytes = limit_bytes
        self.interval = interval
        self._lock = threading.Lock()
        self._sessions: dict[int, dict[str, Any]] = {}  # sid -> {"command", "started"}
        self._killed: dict[int, str] = {}  # sid -> message
        self._background_notices: dict[int, str] = {}  # sid -> notice for the next command
        threading.Thread(target=self._run, daemon=True, name="mswea-memory-watchdog").start()

    def track(self, sid: int, command: str):
        with self._lock:
            self._sessions[sid] = {"command": command, "started": time.monotonic()}

    def killed_message(self, sid: int) -> str | None:
        with self._lock:
            return self._killed.get(sid)

    def foreground_done(self, sid: int):
        """The command's own output reports its kill; don't repeat it as a background notice."""
        with self._lock:
            self._background_notices.pop(sid, None)

    def pop_notices(self) -> list[str]:
        with self._lock:
            notices = list(self._background_notices.values())
            self._background_notices.clear()
        return notices

    def _run(self):
        while True:
            time.sleep(self.interval)
            try:
                self.check()
            except Exception:  # never let the watchdog die
                pass

    def check(self):
        with self._lock:
            tracked = dict(self._sessions)
        if not tracked:
            return
        live = _session_processes()
        for sid, info in tracked.items():
            pids = live.get(sid)
            if not pids:
                with self._lock:
                    self._sessions.pop(sid, None)  # everything it started has exited
                continue
            usage = {pid: _process_memory(pid) for pid in pids}
            total = sum(usage.values())
            if total <= self.limit_bytes:
                continue
            top = max(usage, key=usage.get)
            top_cmd = _process_cmdline(top)
            _killpg(sid)
            elapsed = time.monotonic() - info["started"]
            msg = (
                f"Command killed: memory usage exceeded the {self.limit_bytes / _GIB:g} GB limit "
                f"(reached {total / _GIB:.1f} GB after {elapsed:.0f} s; largest process: "
                f"{top_cmd[:200]!r} at {usage[top] / _GIB:.1f} GB). The command or something it "
                f"started probably has an unbounded loop or allocation; fix it before re-running."
            )
            with self._lock:
                self._sessions.pop(sid, None)
                self._killed[sid] = msg
                self._background_notices[sid] = (
                    f"A background process started by an earlier command ({info['command'][:200]!r}) "
                    f"was killed: {msg}"
                )


class LocalEnvironment:
    def __init__(self, *, config_class: type = LocalEnvironmentConfig, **kwargs):
        """This class executes bash commands directly on the local machine."""
        self.config = config_class(**kwargs)
        self._watchdog = (
            _MemoryWatchdog(int(self.config.memory_limit_gb * _GIB), self.config.memory_poll_interval)
            if self.config.memory_limit_gb
            else None
        )

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict[str, Any]:
        """Execute a command in the local environment and return the result as a dict."""
        command = action.get("command", "")
        cwd = cwd or self.config.cwd or os.getcwd()
        timeout = timeout or self.config.timeout
        if self._watchdog is not None or self.config.kill_process_group_on_timeout:
            output = self._execute_in_session(command, cwd, timeout)
        else:
            output = self._execute_plain(command, cwd, timeout)
        if self._watchdog is not None:
            notices = self._watchdog.pop_notices()
            if notices:
                output["exception_info"] = "\n".join([output["exception_info"], *notices]).strip()
        self._check_finished(output)
        return output

    def _execute_plain(self, command: str, cwd: str, timeout: int) -> dict[str, Any]:
        try:
            result = subprocess.run(
                command,
                shell=True,
                text=True,
                cwd=cwd,
                env=os.environ | self.config.env,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            output = {"output": result.stdout, "returncode": result.returncode, "exception_info": ""}
        except Exception as e:
            raw_output = getattr(e, "output", None)
            raw_output = (
                raw_output.decode("utf-8", errors="replace") if isinstance(raw_output, bytes) else (raw_output or "")
            )
            output = {
                "output": raw_output,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        return output

    def _execute_in_session(self, command: str, cwd: str, timeout: int) -> dict[str, Any]:
        """_execute_plain, but in a new session (= process group) so a timeout or the memory
        watchdog can kill everything the command started. Background processes of a command
        that returns normally are left running, as before."""
        proc = subprocess.Popen(
            command,
            shell=True,
            cwd=cwd,
            env=os.environ | self.config.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        sid = proc.pid
        if self._watchdog is not None:
            self._watchdog.track(sid, command)
        chunks: list[bytes] = []
        reader = threading.Thread(
            target=lambda: chunks.extend(iter(lambda: proc.stdout.read1(65536), b"")), daemon=True
        )
        reader.start()

        deadline = time.monotonic() + timeout
        poll = self.config.memory_poll_interval
        killed_msg = None
        timed_out = False
        # Same completion rule as subprocess.run: the shell exited AND stdout reached EOF (a
        # background process still holding stdout keeps the command running).
        while proc.poll() is None or reader.is_alive():
            if self._watchdog is not None and (killed_msg := self._watchdog.killed_message(sid)):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timed_out = True
                break
            reader.join(min(poll, remaining))

        if timed_out:
            if self.config.kill_process_group_on_timeout:
                _killpg(sid)
            else:
                proc.kill()  # upstream behaviour: only the shell
        if killed_msg or timed_out:
            # The pipe closes once the group is dead. Without the group kill, survivors may
            # hold it open: like subprocess.run, return right away rather than wait for them.
            reader.join(5 if (killed_msg or self.config.kill_process_group_on_timeout) else 0.5)
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                pass
        if self._watchdog is not None:
            # The watchdog may have fired after the last check above (the SIGKILLed shell then
            # simply exits with -9); report it here rather than as a later background notice.
            killed_msg = killed_msg or self._watchdog.killed_message(sid)
            if killed_msg:
                self._watchdog.foreground_done(sid)
        text = b"".join(chunks).decode("utf-8", errors="replace")

        if killed_msg:
            return {
                "output": text,
                "returncode": -1,
                "exception_info": killed_msg,
                "extra": {"exception_type": "MemoryLimitExceeded", "exception": killed_msg},
            }
        if timed_out:
            e = subprocess.TimeoutExpired(command, timeout)
            return {
                "output": text,
                "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {e}",
                "extra": {"exception_type": type(e).__name__, "exception": str(e)},
            }
        return {"output": text, "returncode": proc.returncode, "exception_info": ""}

    def _check_finished(self, output: dict):
        """Raises Submitted if the output indicates task completion."""
        lines = output.get("output", "").lstrip().splitlines(keepends=True)
        if lines and lines[0].strip() == "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" and output["returncode"] == 0:
            submission = "".join(lines[1:])
            raise Submitted(
                {
                    "role": "exit",
                    "content": submission,
                    "extra": {"exit_status": "Submitted", "submission": submission},
                }
            )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return recursive_merge(self.config.model_dump(), platform.uname()._asdict(), os.environ, kwargs)

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "environment": self.config.model_dump(mode="json"),
                    "environment_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                }
            }
        }
