"""
Unit tests for PytestRunner command construction and outcome telemetry parsing.
Iteration 1.5 Slice C.1.
"""

from pathlib import Path
import pytest

from agentic_test.execution.runner import PytestRunner, TestRunSummary


# =====================================================================
# 1. Command Construction Tests
# =====================================================================

def test_construct_command_deterministic_tokens_without_report() -> None:
    """Verifies deterministic CLI token generation without coverage report path."""
    runner = PytestRunner()
    cmd = runner.construct_command(
        tests_path=Path("tests/test_calculator.py"),
        cov_source=Path("src/calculator"),
    )

    expected = [
        "pytest",
        "tests/test_calculator.py",
        "--tb=short",
        "-v",
        "--cov=src/calculator",
        "--cov-branch",
        "--junitxml=/workspace/output/junit.xml",
        "-o",
        "cache_dir=/tmp/.pytest_cache",
    ]
    assert cmd == expected


def test_construct_command_deterministic_tokens_with_report() -> None:
    """Verifies deterministic CLI token generation with JSON coverage report path."""
    runner = PytestRunner()
    cmd = runner.construct_command(
        tests_path="tests",
        cov_source="src",
        cov_report_path="/tmp/coverage.json",
    )

    expected = [
        "pytest",
        "tests",
        "--tb=short",
        "-v",
        "--cov=src",
        "--cov-branch",
        "--junitxml=/workspace/output/junit.xml",
        "--cov-report=json:/tmp/coverage.json",
        "-o",
        "cache_dir=/tmp/.pytest_cache",
    ]
    assert cmd == expected


def test_construct_command_for_paths_includes_every_explicit_test_path() -> None:
    """Regression execution keeps the planner's complete ordered test selection."""
    runner = PytestRunner()

    cmd = runner.construct_command_for_paths(
        tests_paths=(
            "/workspace/tests",
            Path("/workspace/src/tests/test_add.py"),
            Path("/workspace/src/tests/test_mul.py"),
        ),
        cov_source="/workspace/src",
        cov_report_path="/workspace/output/coverage.json",
    )

    assert cmd == [
        "pytest",
        "/workspace/tests",
        "/workspace/src/tests/test_add.py",
        "/workspace/src/tests/test_mul.py",
        "--tb=short",
        "-v",
        "--cov=/workspace/src",
        "--cov-branch",
        "--junitxml=/workspace/output/junit.xml",
        "--cov-report=json:/workspace/output/coverage.json",
        "-o",
        "cache_dir=/tmp/.pytest_cache",
    ]


def test_construct_command_for_paths_rejects_empty_sequence() -> None:
    """An execution command cannot be built without an explicit test target."""
    runner = PytestRunner()

    with pytest.raises(ValueError, match="'tests_paths' cannot be empty"):
        runner.construct_command_for_paths(tests_paths=(), cov_source="src")


def test_construct_command_normalizes_backslashes() -> None:
    """Verifies that Windows backslashes are normalized to POSIX forward slashes."""
    runner = PytestRunner()
    cmd = runner.construct_command(
        tests_path="tests\\unit\\test_foo.py",
        cov_source="src\\pkg\\sub",
        cov_report_path="output\\cov.json",
    )

    assert cmd[1] == "tests/unit/test_foo.py"
    assert cmd[2] == "--tb=short"
    assert cmd[3] == "-v"
    assert cmd[4] == "--cov=src/pkg/sub"
    assert cmd[5] == "--cov-branch"
    assert cmd[6] == "--junitxml=/workspace/output/junit.xml"
    assert cmd[7] == "--cov-report=json:output/cov.json"


def test_construct_command_rejects_empty_paths() -> None:
    """Verifies that empty string or whitespace-only paths raise ValueError."""
    runner = PytestRunner()

    with pytest.raises(ValueError, match="'tests_path' cannot be empty"):
        runner.construct_command(tests_path="", cov_source="src")

    with pytest.raises(ValueError, match="'cov_source' cannot be empty"):
        runner.construct_command(tests_path="tests", cov_source="   ")

    with pytest.raises(ValueError, match="'cov_report_path' cannot be empty"):
        runner.construct_command(
            tests_path="tests", cov_source="src", cov_report_path=""
        )


def test_construct_command_rejects_path_traversal() -> None:
    """Verifies that paths containing directory traversal ('..') are strictly rejected."""
    runner = PytestRunner()

    with pytest.raises(ValueError, match="Path traversal .* not permitted"):
        runner.construct_command(tests_path="../tests", cov_source="src")

    with pytest.raises(ValueError, match="Path traversal .* not permitted"):
        runner.construct_command(tests_path="tests", cov_source="src/../../etc")

    with pytest.raises(ValueError, match="Path traversal .* not permitted"):
        runner.construct_command(
            tests_path="tests",
            cov_source="src",
            cov_report_path="/tmp/../etc/passwd",
        )


def test_construct_command_rejects_windows_drive_letters() -> None:
    """Verifies that paths with Windows drive letters are rejected for container execution."""
    runner = PytestRunner()

    with pytest.raises(ValueError, match="Windows drive-letter paths are not permitted"):
        runner.construct_command(tests_path="C:/tests", cov_source="src")

    with pytest.raises(ValueError, match="Windows drive-letter paths are not permitted"):
        runner.construct_command(tests_path="tests", cov_source="D:\\project\\src")


# =====================================================================
# 2. Outcome Parsing Tests: Success and Skip Scenarios
# =====================================================================

def test_parse_test_outcomes_all_passed() -> None:
    """Verifies parsing of clean session where all tests passed."""
    stdout = """
============================= test session starts ==============================
platform linux -- Python 3.11.16, pytest-8.4.2, pluggy-1.6.0
rootdir: /workspace
collected 5 items

tests/test_math.py .....                                                 [100%]

============================== 5 passed in 0.25s ===============================
"""
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout=stdout, stderr="", exit_code=0)

    assert summary.exit_code == 0
    assert summary.passed == 5
    assert summary.failed == 0
    assert summary.errors == 0
    assert summary.skipped == 0
    assert summary.duration_sec == 0.25
    assert summary.timed_out is False
    assert summary.oom_killed is False
    assert summary.traceback is None
    assert summary.success is True
    assert "All 5 test(s) passed" in summary.status_message


def test_parse_test_outcomes_passed_and_skipped() -> None:
    """Verifies parsing of mixed passed and skipped tests with exit code 0."""
    stdout = """
============================= test session starts ==============================
rootdir: /workspace
collected 4 items

tests/test_mod.py ..ss                                                   [100%]

========================= 2 passed, 2 skipped in 0.18s =========================
"""
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout=stdout, stderr="", exit_code=0)

    assert summary.exit_code == 0
    assert summary.passed == 2
    assert summary.skipped == 2
    assert summary.failed == 0
    assert summary.errors == 0
    assert summary.duration_sec == 0.18
    assert summary.success is True
    assert "2 test(s) passed, 2 skipped" in summary.status_message


def test_parse_test_outcomes_all_skipped() -> None:
    """Verifies parsing when all collected tests are skipped."""
    stdout = """
============================= test session starts ==============================
rootdir: /workspace
collected 3 items

tests/test_skip.py sss                                                   [100%]

============================== 3 skipped in 0.05s ==============================
"""
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout=stdout, stderr="", exit_code=0)

    assert summary.exit_code == 0
    assert summary.passed == 0
    assert summary.skipped == 3
    assert summary.failed == 0
    assert summary.errors == 0
    assert summary.duration_sec == 0.05
    assert summary.success is True
    assert "All 3 test(s) skipped" in summary.status_message


# =====================================================================
# 3. Outcome Parsing Tests: Failures and Traceback Extraction
# =====================================================================

def test_parse_test_outcomes_failure_with_traceback() -> None:
    """Verifies failure count, exit code 1, and assertion traceback extraction."""
    stdout = """
============================= test session starts ==============================
rootdir: /workspace
collected 2 items

tests/test_calc.py .F                                                    [100%]

=================================== FAILURES ===================================
_________________________________ test_divide __________________________________

    def test_divide():
>       assert 10 / 2 == 4
E       assert 5.0 == 4

tests/test_calc.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_calc.py::test_divide - assert 5.0 == 4
========================= 1 failed, 1 passed in 0.12s ==========================
"""
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout=stdout, stderr="", exit_code=1)

    assert summary.exit_code == 1
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.errors == 0
    assert summary.duration_sec == 0.12
    assert summary.success is False
    assert "1 test(s) failed" in summary.status_message
    assert summary.traceback is not None
    assert "test_divide" in summary.traceback
    assert "assert 10 / 2 == 4" in summary.traceback
    assert "AssertionError" in summary.traceback


def test_parse_test_outcomes_collection_error() -> None:
    """Verifies collection error extraction from ERRORS block and exit code 1."""
    stdout = """
============================= test session starts ==============================
rootdir: /workspace
collected 0 items / 1 error

==================================== ERRORS ====================================
____________________ ERROR collecting tests/test_missing.py ____________________
ImportError while importing test module '/workspace/tests/test_missing.py'.
Traceback:
/usr/local/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
E   ModuleNotFoundError: No module named 'nonexistent_package'
=========================== short test summary info ============================
ERROR tests/test_missing.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
=============================== 1 error in 0.45s ===============================
"""
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout=stdout, stderr="", exit_code=1)

    assert summary.exit_code == 1
    assert summary.passed == 0
    assert summary.failed == 0
    assert summary.errors == 1
    assert summary.duration_sec == 0.45
    assert summary.success is False
    assert "1 error(s)" in summary.status_message
    assert summary.traceback is not None
    assert "ModuleNotFoundError: No module named 'nonexistent_package'" in summary.traceback


def test_parse_test_outcomes_mixed_failures_errors_and_passes() -> None:
    """Verifies parsing when stdout reports passes, failures, skipped, and errors."""
    stdout = """
=================================== FAILURES ===================================
__________________________________ test_fail ___________________________________
    def test_fail():
>       assert False
E       assert False
========================= 2 failed, 4 passed, 1 skipped, 1 error in 1.25s =========================
"""
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout=stdout, stderr="", exit_code=1)

    assert summary.exit_code == 1
    assert summary.passed == 4
    assert summary.failed == 2
    assert summary.skipped == 1
    assert summary.errors == 1
    assert summary.duration_sec == 1.25
    assert summary.success is False
    assert "2 test(s) failed, 1 error(s)" in summary.status_message
    assert summary.traceback is not None
    assert "test_fail" in summary.traceback


# =====================================================================
# 4. Outcome Parsing Tests: Special Exit Codes (2, 5, 124, 137, Unknown)
# =====================================================================

def test_parse_test_outcomes_interrupted_session_exit_2() -> None:
    """Verifies handling of user or signal interruption (exit code 2)."""
    stdout = """
============================= test session starts ==============================
rootdir: /workspace
collected 10 items

tests/test_big.py ..
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! KeyboardInterrupt !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout=stdout, stderr="", exit_code=2)

    assert summary.exit_code == 2
    assert summary.timed_out is False
    assert summary.oom_killed is False
    assert summary.success is False
    assert "interrupted" in summary.status_message.lower()


def test_parse_test_outcomes_no_tests_collected_exit_5() -> None:
    """Verifies handling of exit code 5 (no tests found)."""
    stdout = """
============================= test session starts ==============================
rootdir: /workspace
collected 0 items

============================ no tests ran in 0.01s =============================
"""
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout=stdout, stderr="", exit_code=5)

    assert summary.exit_code == 5
    assert summary.passed == 0
    assert summary.failed == 0
    assert summary.duration_sec == 0.01
    assert summary.success is False
    assert "no tests were collected" in summary.status_message.lower()


def test_parse_test_outcomes_timeout_exit_124() -> None:
    """Verifies handling of host watchdog timeout (exit code 124)."""
    stderr = "Command execution timed out after 30.0s (SIGKILL)"
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout="", stderr=stderr, exit_code=124)

    assert summary.exit_code == 124
    assert summary.timed_out is True
    assert summary.oom_killed is False
    assert summary.success is False
    assert "timed out" in summary.status_message.lower()
    assert summary.traceback == stderr


def test_parse_test_outcomes_oom_killed_exit_137() -> None:
    """Verifies handling of container OOM kill (exit code 137)."""
    stderr = "Killed: Out of memory"
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout="", stderr=stderr, exit_code=137)

    assert summary.exit_code == 137
    assert summary.oom_killed is True
    assert summary.timed_out is False
    assert summary.success is False
    assert "out-of-memory" in summary.status_message.lower()
    assert summary.traceback == stderr


def test_parse_test_outcomes_traceback_in_stderr() -> None:
    """Verifies traceback extraction from stderr when stdout has no failure block."""
    stderr = """
Traceback (most recent call last):
  File "/workspace/tests/test_fatal.py", line 4, in <module>
    import missing_native_lib
SystemError: initialization failed
"""
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout="", stderr=stderr, exit_code=1)

    assert summary.exit_code == 1
    assert summary.traceback is not None
    assert "SystemError: initialization failed" in summary.traceback
    assert "missing_native_lib" in summary.traceback


def test_parse_test_outcomes_malformed_incomplete_output() -> None:
    """Verifies handling of truncated/malformed output with non-standard exit code."""
    stdout = "Fatal Python error: Segmentation fault"
    runner = PytestRunner()
    summary = runner.parse_test_outcomes(stdout=stdout, stderr="", exit_code=139)

    assert summary.exit_code == 139
    assert summary.passed == 0
    assert summary.failed == 0
    assert summary.duration_sec == 0.0
    assert summary.success is False
    assert "exit code 139" in summary.status_message
    assert summary.traceback == stdout
