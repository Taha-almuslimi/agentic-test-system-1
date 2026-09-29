"""
Gate 3 pytest collection validation inside an isolated sandbox.

Candidate modules execute import-time code during pytest collection.  This
validator therefore never invokes pytest on the host: it stages one candidate
in a temporary directory and asks the SandboxManager protocol to collect it in
the hardened runner container.
"""

from pathlib import Path
import tempfile
from typing import Optional, Tuple

from agentic_test.core.models import TestCandidate, ValidationStatus
from agentic_test.core.protocols.sandbox import SandboxConfig, SandboxError, SandboxManager
from agentic_test.validation.base import BaseValidator
from agentic_test.validation.models import ValidationResult


class CollectionValidator(BaseValidator):
    """Runs the Gate 3 dry-run collection command in an isolated container."""

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        source_root: Path,
        collection_staging_root: Optional[Path] = None,
        timeout_sec: float = 30.0,
    ) -> None:
        if timeout_sec <= 0:
            raise ValueError("Collection timeout_sec must be positive.")

        self._sandbox_manager = sandbox_manager
        self._source_root = source_root.resolve()
        if not self._source_root.is_dir():
            raise ValueError(
                f"Collection source_root must be an existing repository directory: "
                f"{self._source_root}"
            )

        self._collection_staging_root = (
            collection_staging_root.resolve() if collection_staging_root else None
        )
        if (
            self._collection_staging_root is not None
            and not self._collection_staging_root.is_dir()
        ):
            raise ValueError(
                "collection_staging_root must be an existing directory when provided: "
                f"{self._collection_staging_root}"
            )
        self._timeout_sec = timeout_sec

    @property
    def gate_name(self) -> str:
        return "GATE_3_COLLECTION"

    @staticmethod
    def _format_candidate_artifact(candidate: TestCandidate) -> str:
        """Builds the temporary test module without using its output path."""
        sections = []
        imports = [import_line.strip() for import_line in candidate.imports if import_line.strip()]
        if imports:
            sections.append("\n".join(imports))

        code = candidate.candidate_code.strip()
        if code:
            sections.append(code)
        return "\n\n".join(sections) + "\n"

    @staticmethod
    def _diagnostics(stdout: str, stderr: str) -> Tuple[str, ...]:
        return tuple(output for output in (stdout.strip(), stderr.strip()) if output)

    def _rejection(
        self,
        candidate: TestCandidate,
        error_message: str,
        diagnostics: Tuple[str, ...] = (),
    ) -> ValidationResult:
        return ValidationResult(
            candidate_id=candidate.candidate_id,
            status=ValidationStatus.REJECTED_COLLECTION,
            gate=self.gate_name,
            passed=False,
            error_message=error_message,
            diagnostics=diagnostics,
        )

    def validate(self, candidate: TestCandidate) -> ValidationResult:
        """Collects a staged candidate in Docker and reports its collection result."""
        container_id: Optional[str] = None
        try:
            with tempfile.TemporaryDirectory(
                prefix="agentic-collection-",
                dir=self._collection_staging_root,
            ) as temporary_root:
                tests_dir = Path(temporary_root) / "tests"
                tests_dir.mkdir()
                (tests_dir / "test_candidate.py").write_text(
                    self._format_candidate_artifact(candidate),
                    encoding="utf-8",
                )

                sandbox_config = SandboxConfig(
                    timeout_sec=self._timeout_sec,
                    read_only_mounts={
                        self._source_root: "/workspace/src",
                        tests_dir: "/workspace/tests",
                    },
                )
                container_id = self._sandbox_manager.create_environment(sandbox_config)
                try:
                    raw_result = self._sandbox_manager.execute_command(
                        container_id=container_id,
                        command=[
                            "pytest",
                            "--collect-only",
                            "/workspace/tests",
                            "-o",
                            "cache_dir=/tmp/.pytest_cache",
                        ],
                        workdir="/workspace",
                    )
                finally:
                    self._sandbox_manager.cleanup(container_id)
                    container_id = None
        except (OSError, RuntimeError, SandboxError, ValueError) as exc:
            if container_id is not None:
                try:
                    self._sandbox_manager.cleanup(container_id)
                except SandboxError:
                    pass
            return self._rejection(
                candidate,
                f"Pytest collection could not run in the sandbox: {exc}",
            )

        diagnostics = self._diagnostics(raw_result.stdout, raw_result.stderr)
        if raw_result.exit_code == 0 and not raw_result.timed_out and not raw_result.oom_killed:
            return ValidationResult(
                candidate_id=candidate.candidate_id,
                status=ValidationStatus.PASSED,
                gate=self.gate_name,
                passed=True,
                diagnostics=diagnostics,
            )

        if raw_result.timed_out:
            message = "Pytest collection timed out in the sandbox."
        elif raw_result.oom_killed:
            message = "Pytest collection exceeded sandbox resources."
        else:
            message = f"Pytest collection failed with exit code {raw_result.exit_code}."
        return self._rejection(candidate, message, diagnostics)
