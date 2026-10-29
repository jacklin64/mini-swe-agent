"""LocalEnvironment memory_limit_gb / kill_process_group_on_timeout (Linux only: reads /proc)."""

import os
import sys
import textwrap
import time
import uuid

import pytest

from minisweagent.environments.local import LocalEnvironment

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="needs /proc")

LIMIT_GB = 0.3


def _grow_script(tmp_path, tag: str, step_mb: int = 40, max_mb: int = 3000, delay: float = 0.05):
    """A python script that keeps allocating (and touching) memory -- the check_spec.py shape."""
    p = tmp_path / f"grow_{tag}.py"
    p.write_text(
        textwrap.dedent(f"""
            import time, sys
            hog = []
            for i in range({max_mb // step_mb}):
                hog.append(b"x" * ({step_mb} * 1024 * 1024))
                print("allocated", (i + 1) * {step_mb}, "MB", flush=True)
                time.sleep({delay})
            print("FINISHED_WITHOUT_BEING_KILLED", flush=True)
        """)
    )
    return p


def _alive(pattern: str) -> bool:
    for entry in os.listdir("/proc"):
        if entry.isdigit():
            try:
                with open(f"/proc/{entry}/cmdline", "rb") as f:
                    if pattern.encode() in f.read():
                        return True
            except OSError:
                pass
    return False


def _wait_gone(pattern: str, secs: float = 5) -> bool:
    deadline = time.monotonic() + secs
    while time.monotonic() < deadline:
        if not _alive(pattern):
            return True
        time.sleep(0.1)
    return False


def test_normal_command_unaffected(tmp_path):
    env = LocalEnvironment(memory_limit_gb=LIMIT_GB, kill_process_group_on_timeout=True, timeout=30)
    out = env.execute({"command": "echo hello; echo err >&2; exit 3"}, cwd=str(tmp_path))
    assert out["returncode"] == 3
    assert "hello" in out["output"] and "err" in out["output"]
    assert out["exception_info"] == ""


def test_runaway_is_killed_and_reported(tmp_path):
    script = _grow_script(tmp_path, "fg")
    env = LocalEnvironment(memory_limit_gb=LIMIT_GB, timeout=120)
    t0 = time.monotonic()
    out = env.execute({"command": f"python3 {script}"}, cwd=str(tmp_path))
    assert time.monotonic() - t0 < 30
    assert out["returncode"] == -1
    assert out["extra"]["exception_type"] == "MemoryLimitExceeded"
    assert "memory usage exceeded the 0.3 GB limit" in out["exception_info"]
    assert str(script.name) in out["exception_info"]  # names the offending process
    assert "allocated" in out["output"]  # partial output kept
    assert "FINISHED_WITHOUT_BEING_KILLED" not in out["output"]
    assert _wait_gone(script.name)


def test_runaway_behind_pipe_is_killed_entirely(tmp_path):
    """Agent-style `cmd 2>&1 | tail`: the python grandchild must die too, not just sh."""
    script = _grow_script(tmp_path, "pipe")
    env = LocalEnvironment(memory_limit_gb=LIMIT_GB, timeout=120)
    out = env.execute({"command": f"python3 {script} 2>&1 | tail -3"}, cwd=str(tmp_path))
    assert out["extra"]["exception_type"] == "MemoryLimitExceeded"
    assert _wait_gone(script.name)


def test_timeout_kills_whole_group(tmp_path):
    """The test_supervisor.py case: timeout must not leave the pipeline's producer running."""
    tag = uuid.uuid4().hex[:8]
    env = LocalEnvironment(kill_process_group_on_timeout=True, timeout=2)
    # The tag rides along as an ignored argv entry so the sleeper can be found in /proc.
    out = env.execute(
        {"command": f"echo start; python3 -c 'import time; time.sleep(300)' {tag} | tail -1"}, cwd=str(tmp_path)
    )
    assert out["returncode"] == -1
    assert out["extra"]["exception_type"] == "TimeoutExpired"
    assert "timed out after 2 seconds" in out["exception_info"]
    assert "start" in out["output"]
    assert _wait_gone(tag)


def test_timeout_without_group_kill_keeps_upstream_behaviour(tmp_path):
    """Documents the old bug: with the option off, the pipeline producer survives the timeout."""
    tag = uuid.uuid4().hex[:8]
    env = LocalEnvironment(memory_limit_gb=LIMIT_GB, kill_process_group_on_timeout=False, timeout=2)
    out = env.execute({"command": f"python3 -c 'import time; time.sleep(30)' {tag} | tail -1"}, cwd=str(tmp_path))
    try:
        assert out["extra"]["exception_type"] == "TimeoutExpired"
        assert _alive(tag)
    finally:
        os.system(f"pkill -f {tag}")


def test_slow_orphan_after_timeout_is_still_policed(tmp_path):
    """Timeout without group kill leaves the producer running; the watchdog still catches it
    when it grows over the limit, and the next command is told."""
    # Slow enough (~12 s to the limit) that it is still under it when the timed-out command
    # returns, so the kill shows up as a notice on the *next* command.
    script = _grow_script(tmp_path, "orphan", step_mb=10, delay=0.4)
    env = LocalEnvironment(memory_limit_gb=LIMIT_GB, kill_process_group_on_timeout=False, timeout=2)
    out = env.execute({"command": f"python3 {script} | tail -1"}, cwd=str(tmp_path))
    assert out["extra"]["exception_type"] == "TimeoutExpired"
    assert _wait_gone(script.name, secs=30)
    out = env.execute({"command": "echo next"}, cwd=str(tmp_path))
    assert out["returncode"] == 0 and "next" in out["output"]
    assert "A background process started by an earlier command" in out["exception_info"]
    assert "memory usage exceeded" in out["exception_info"]


def test_background_runaway_killed_between_commands(tmp_path):
    script = _grow_script(tmp_path, "bg", step_mb=20, delay=0.2)
    env = LocalEnvironment(memory_limit_gb=LIMIT_GB, timeout=30)
    out = env.execute({"command": f"nohup python3 {script} >/dev/null 2>&1 & echo launched"}, cwd=str(tmp_path))
    assert out["returncode"] == 0 and "launched" in out["output"] and out["exception_info"] == ""
    assert _wait_gone(script.name, secs=30)  # killed while no command is running
    out = env.execute({"command": "echo next"}, cwd=str(tmp_path))
    assert "A background process started by an earlier command" in out["exception_info"]
    out = env.execute({"command": "echo again"}, cwd=str(tmp_path))
    assert out["exception_info"] == ""  # reported once


def test_benign_background_process_survives(tmp_path):
    tag = uuid.uuid4().hex[:8]
    env = LocalEnvironment(memory_limit_gb=LIMIT_GB, kill_process_group_on_timeout=True, timeout=10)
    try:
        out = env.execute(
            {"command": f"nohup python3 -c 'import time; time.sleep(60)' {tag} >/dev/null 2>&1 & echo ok"},
            cwd=str(tmp_path),
        )
        assert out["returncode"] == 0
        time.sleep(1.5)
        out = env.execute({"command": f"pgrep -f {tag} >/dev/null && echo still-running"}, cwd=str(tmp_path))
        assert "still-running" in out["output"]
    finally:
        os.system(f"pkill -f {tag}")


@pytest.mark.parametrize(
    "command",
    [
        "echo a; echo b >&2; exit 0",
        "(sleep 1; echo late) & echo early",  # background still holding stdout: wait for EOF
        "printf 'x%.0s' $(seq 1 200000) | wc -c",
        "exit 7",
    ],
)
def test_session_mode_matches_plain_mode(tmp_path, command):
    plain = LocalEnvironment(timeout=20).execute({"command": command}, cwd=str(tmp_path))
    session = LocalEnvironment(memory_limit_gb=LIMIT_GB, kill_process_group_on_timeout=True, timeout=20).execute(
        {"command": command}, cwd=str(tmp_path)
    )
    assert session["returncode"] == plain["returncode"]
    assert session["output"] == plain["output"]
    assert session["exception_info"] == plain["exception_info"] == ""
