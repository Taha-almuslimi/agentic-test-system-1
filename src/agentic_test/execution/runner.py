"""
Test execution command constructor and pytest outcome telemetry parser.
Stage 3 Section 4.4.4.5 Listing 4.2.
Iteration 1.5 Slice C.1.
"""

from pathlib import Path
import re
from typing import List, Optional, Sequence, Union
from pydantic import BaseModel, ConfigDict


class TestRunSummary(BaseModel):
    """
    Structured test execution outcome telemetry parsed from pytest stdout/stderr.
    Stage 3 Section 4.4.4.5.
    """
    __test__ = False
    model_config = ConfigDict(frozen=True)

    exit_code: int
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    duration_sec: float = 0.0
    timed_out: bool = False
    oom_killed: bool = False
    traceback: Optional[str] = None
    status_message: str = ""

    @property
    def success(self) -> bool:
        """Returns True if the execution completed with zero failures or errors."""
        return self.exit_code == 0 and self.failed == 0 and self.errors == 0


class PytestRunner:
    """
    Pure Python test runner command constructor and outcome telemetry parser.
    Constructs deterministic pytest CLI tokens and parses execution outcomes
    without spawning subprocesses or interacting with the container runtime.
    Stage 3 Section 4.4.4.5.
    """

    _JUNIT_REPORT_PATH = "/workspace/output/junit.xml"

    def construct_command(
        self,
        tests_path: Union[Path, str],
        cov_source: Union[Path, str],
        cov_report_path: Optional[Union[Path, str]] = None,
    ) -> List[str]:
        """
        Constructs deterministic pytest CLI invocation arguments for container execution.

        :param tests_path: Path to target test file or directory within container.
        :param cov_source: Path to source code root for coverage instrumentation.
        :param cov_report_path: Optional path for JSON coverage report output.
        :return: Deterministic list of CLI command tokens.
        :raises ValueError: If any path is empty, contains directory traversal ('..'),
                            or specifies a Windows drive letter.
        """
        clean_tests = self._validate_and_normalize_path(tests_path, "tests_path")
        clean_cov = self._validate_and_normalize_path(cov_source, "cov_source")

        tokens = [
            "pytest",
            clean_tests,
            "--tb=short",
            "-v",
            f"--cov={clean_cov}",
            "--cov-branch",
            f"--junitxml={self._JUNIT_REPORT_PATH}",
        ]

        if cov_report_path is not None:
            clean_report = self._validate_and_normalize_path(
                cov_report_path, "cov_report_path"
            )
            tokens.append(f"--cov-report=json:{clean_report}")

        tokens.extend(["-o", "cache_dir=/tmp/.pytest_cache"])
        return tokens

    def construct_command_for_paths(
        self,
        tests_paths: Sequence[Union[Path, str]],
        cov_source: Union[Path, str],
        cov_report_path: Optional[Union[Path, str]] = None,
    ) -> List[str]:
        """
        Constructs one pytest command for several explicit test paths.

        :param tests_paths: Non-empty ordered test file or directory paths inside the container.
        :param cov_source: Path to source code root for coverage instrumentation.
        :param cov_report_path: Optional path for JSON coverage report output.
        :return: Deterministic list of CLI command tokens.
        :raises ValueError: If no test paths are supplied or any supplied path is unsafe.
        """
        if not tests_paths:
            raise ValueError("'tests_paths' cannot be empty")

        clean_tests = [
            self._validate_and_normalize_path(test_path, f"tests_paths[{index}]")
            for index, test_path in enumerate(tests_paths)
        ]
        clean_cov = self._validate_and_normalize_path(cov_source, "cov_source")
        tokens = [
            "pytest",
            *clean_tests,
            "--tb=short",
            "-v",
            f"--cov={clean_cov}",
            "--cov-branch",
            f"--junitxml={self._JUNIT_REPORT_PATH}",
        ]

        if cov_report_path is not None:
            clean_report = self._validate_and_normalize_path(
                cov_report_path, "cov_report_path"
            )
            tokens.append(f"--cov-report=json:{clean_report}")

        tokens.extend(["-o", "cache_dir=/tmp/.pytest_cache"])
        return tokens

    def parse_test_outcomes(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> TestRunSummary:
        """
        Parses raw pytest stdout, stderr, and exit code into a structured TestRunSummary.

        :param stdout: Captured standard output from test run.
        :param stderr: Captured standard error from test run.
        :param exit_code: Process exit code.
        :return: TestRunSummary containing counts, timings, and diagnostic details.
        """
        norm_stdout = stdout or ""
        norm_stderr = stderr or ""

        # 1. Detect timeout and OOM states
        timed_out = (exit_code == 124) or ("timed out" in norm_stderr.lower())
        oom_killed = (exit_code == 137) or ("out of memory" in norm_stderr.lower())

        # 2. Extract counts and duration from pytest summary lines
        passed = 0
        failed = 0
        errors = 0
        skipped = 0
        duration_sec = 0.0

        summary_line = self._find_summary_line(norm_stdout)
        if summary_line:
            duration_m = re.search(r"in\s+([0-9]+(?:\.[0-9]+)?)\s*s", summary_line)
            if duration_m:
                try:
                    duration_sec = float(duration_m.group(1))
                except ValueError:
                    duration_sec = 0.0

            passed_m = re.search(r"\b([0-9]+)\s+passed\b", summary_line)
            if passed_m:
                passed = int(passed_m.group(1))

            failed_m = re.search(r"\b([0-9]+)\s+failed\b", summary_line)
            if failed_m:
                failed = int(failed_m.group(1))

            errors_m = re.search(r"\b([0-9]+)\s+errors?\b", summary_line)
            if errors_m:
                errors = int(errors_m.group(1))

            skipped_m = re.search(r"\b([0-9]+)\s+skipped\b", summary_line)
            if skipped_m:
                skipped = int(skipped_m.group(1))

        # Fallback collection error checks if no summary line matched
        if errors == 0:
            coll_err_m = re.search(
                r"Interrupted:\s+([0-9]+)\s+errors?\s+during\s+collection",
                norm_stdout,
            )
            if coll_err_m:
                errors = int(coll_err_m.group(1))
            else:
                coll_items_m = re.search(
                    r"collected\s+0\s+items\s+/\s+([0-9]+)\s+errors?",
                    norm_stdout,
                )
                if coll_items_m:
                    errors = int(coll_items_m.group(1))

        # 3. Extract failure traceback / diagnostic details
        traceback_text = self._extract_traceback(norm_stdout, norm_stderr, exit_code)

        # 4. Synthesize diagnostic status message
        status_message = self._interpret_status_message(
            exit_code=exit_code,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            timed_out=timed_out,
            oom_killed=oom_killed,
        )

        return TestRunSummary(
            exit_code=exit_code,
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            duration_sec=duration_sec,
            timed_out=timed_out,
            oom_killed=oom_killed,
            traceback=traceback_text,
            status_message=status_message,
        )

    @staticmethod
    def _validate_and_normalize_path(path_val: Union[Path, str], field_name: str) -> str:
        """
        Validates path security constraints and formats as POSIX string for container.
        """
        raw_str = str(path_val).strip()
        if not raw_str:
            raise ValueError(f"'{field_name}' cannot be empty")

        # Disallow Windows drive letter specifications inside container commands
        if re.match(r"^[a-zA-Z]:", raw_str):
            raise ValueError(
                f"Windows drive-letter paths are not permitted in container commands for '{field_name}': {raw_str}"
            )

        # Normalize backslashes to forward slashes
        posix_str = raw_str.replace("\\", "/")

        # Disallow directory traversal segments
        parts = [seg for seg in posix_str.split("/") if seg]
        if ".." in parts:
            raise ValueError(
                f"Path traversal ('..') is not permitted in '{field_name}': {raw_str}"
            )

        return posix_str

    @staticmethod
    def _find_summary_line(stdout: str) -> Optional[str]:
        """Searches backward from end of stdout to locate pytest's summary line."""
        for line in reversed(stdout.splitlines()):
            if re.search(r"in\s+[0-9]+(?:\.[0-9]+)?\s*s", line):
                return line
        return None

    @staticmethod
    def _extract_traceback(stdout: str, stderr: str, exit_code: int) -> Optional[str]:
        """
        Extracts diagnostic traceback or failure messages from stdout and stderr.
        """
        # A. Look for standard pytest FAILURES or ERRORS block in stdout
        tb_match = re.search(
            r"(=+\s*(?:FAILURES|ERRORS)\s*=+\n[\s\S]*?)(?==+\s*short test summary info\s*=+|=+\s*[\d\w\s,]+ in [\d\.]+s\s*=+|\Z)",
            stdout,
        )
        if tb_match:
            raw_chunk = tb_match.group(1).strip()
            lines = raw_chunk.splitlines()
            if lines and ("FAILURES" in lines[0] or "ERRORS" in lines[0]):
                inner = "\n".join(lines[1:]).strip()
                if inner:
                    return inner
            return raw_chunk

        # B. Look for standard Python Traceback in stdout
        if "Traceback (most recent call last):" in stdout:
            py_tb_match = re.search(
                r"(Traceback \(most recent call last\):[\s\S]*?)(?=\n=+|\Z)",
                stdout,
            )
            if py_tb_match:
                return py_tb_match.group(1).strip()

        # C. Look for standard Python Traceback in stderr
        if "Traceback (most recent call last):" in stderr:
            err_tb_match = re.search(
                r"(Traceback \(most recent call last\):[\s\S]*?)(?=\n=+|\Z)",
                stderr,
            )
            if err_tb_match:
                return err_tb_match.group(1).strip()

        # D. Look for short test summary info section in stdout
        if "= short test summary info =" in stdout:
            summary_info_match = re.search(
                r"=+\s*short test summary info\s*=+\n([\s\S]*?)(?==+\s*[\d\w\s,]+ in [\d\.]+s\s*=+|\Z)",
                stdout,
            )
            if summary_info_match:
                content = summary_info_match.group(1).strip()
                if content:
                    return content

        # E. Fallback for non-zero execution with stderr or stdout content
        if exit_code != 0:
            if stderr.strip():
                return stderr.strip()
            if stdout.strip():
                # Provide tail of stdout if no structured traceback was extracted
                tail_lines = stdout.strip().splitlines()[-10:]
                return "\n".join(tail_lines).strip()

        return None

    @staticmethod
    def _interpret_status_message(
        exit_code: int,
        passed: int,
        failed: int,
        errors: int,
        skipped: int,
        timed_out: bool,
        oom_killed: bool,
    ) -> str:
        """Synthesizes human-readable status message for given execution state."""
        if timed_out or exit_code == 124:
            return "Command execution timed out (SIGKILL)"
        if oom_killed or exit_code == 137:
            return "Process terminated due to out-of-memory (OOM) or SIGKILL"
        if exit_code == 2:
            return "Execution interrupted by user or signal"
        if exit_code == 5:
            return "No tests were collected"
        if exit_code == 0:
            if passed > 0 and skipped == 0:
                return f"All {passed} test(s) passed"
            if passed > 0 and skipped > 0:
                return f"{passed} test(s) passed, {skipped} skipped"
            if passed == 0 and skipped > 0:
                return f"All {skipped} test(s) skipped"
            return "Test session passed"
        if exit_code == 1:
            parts: List[str] = []
            if failed > 0:
                parts.append(f"{failed} test(s) failed")
            if errors > 0:
                parts.append(f"{errors} error(s)")
            if parts:
                return ", ".join(parts)
            return "Test session failed"

        return f"Pytest exited with non-zero exit code {exit_code}"
