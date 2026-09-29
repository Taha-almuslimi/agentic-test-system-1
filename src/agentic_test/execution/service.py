"""
ExecutionService orchestrating sandboxed container execution and telemetry extraction.
Stage 2 Section 4.3.2.5 (UC-05), Stage 3 Section 4.4.4.5 Listing 4.2.
Iteration 1.5 Slice D.
"""

from datetime import datetime, timezone
import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import Dict, List, Optional, Sequence, Union
import uuid

from agentic_test.analysis.git_service import is_excluded_path
from agentic_test.core.models import (
    ExecutionEvidence,
    ExecutionPlan,
    TestCandidate,
    ValidationStatus,
)
from agentic_test.core.protocols.sandbox import (
    SandboxConfig,
    SandboxManager,
)
from agentic_test.execution.coverage import CoverageExtractor, CoverageMetrics
from agentic_test.execution.runner import PytestRunner, TestRunSummary
from agentic_test.execution.staging import CandidateStagingArea, StagingValidationError


class ExecutionError(Exception):
    """Base exception for execution subsystem errors."""
    pass


class SourceIntegrityError(ExecutionError):
    """
    Raised when Safety Invariant INV-02 is violated:
    Production source files were modified during sandbox execution.
    UC-05 Extension 9a: PRODUCTION_SOURCE_INTEGRITY_VIOLATION.
    """
    pass


# Formal Stage 2 alias for UC-05 Extension 9a
ProductionSourceIntegrityViolation = SourceIntegrityError


def compute_tree_hash(root: Path) -> str:
    """
    Computes a deterministic cryptographic SHA-256 hash of all files under root.
    Excludes caches, virtualenvs, and .git directories (INV-02).

    :param root: Root directory path to hash.
    :return: 64-character lowercase hex digest string.
    """
    hasher = hashlib.sha256()
    if not root.exists():
        return hasher.hexdigest()

    if root.is_file():
        try:
            hasher.update(root.read_bytes())
        except OSError:
            pass
        return hasher.hexdigest()

    # Collect non-excluded files in deterministic sorted order
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            try:
                rel = p.relative_to(root)
            except ValueError:
                continue
            if not is_excluded_path(rel):
                files.append(p)

    files.sort()
    for f in files:
        try:
            rel_str = f.relative_to(root).as_posix()
            content_hash = hashlib.sha256(f.read_bytes()).hexdigest()
            hasher.update(f"{rel_str}:{content_hash}\n".encode("utf-8"))
        except OSError:
            continue

    return hasher.hexdigest()


class ExecutionService:
    """
    High-level facade coordinating container sandbox execution and evidence capture.
    Stage 2 UC-05, Stage 3 Section 4.4.4.5.

    Coordinates:
    1. Validation filtering & candidate staging to isolated ephemeral mount (INV-04).
    2. Pre/post cryptographic SHA-256 source tree hashing (INV-02).
    3. Deterministic CLI command construction via PytestRunner.
    4. Ephemeral container lifecycle confinement via SandboxManager (INV-01).
    5. Test outcome telemetry parsing & failure traceback extraction.
    6. Coverage telemetry extraction via CoverageExtractor.
    7. Fail-closed cleanup of container, staging volume, and scratch mounts in all paths.
    """

    def __init__(
        self,
        sandbox_manager: SandboxManager,
        runner: Optional[PytestRunner] = None,
        coverage_extractor: Optional[CoverageExtractor] = None,
        source_root: Optional[Path] = None,
        staging_root: Optional[Path] = None,
        scratch_root: Optional[Path] = None,
        timeout_sec: float = 30.0,
        image_tag: str = "agentic-runner:0.1.0",
        memory_limit: str = "512m",
        cpu_quota: float = 1.0,
        pids_limit: int = 100,
        cov_source: str = "/workspace/src",
        environment_vars: Optional[Dict[str, str]] = None,
        keep_artifacts: bool = False,
    ) -> None:
        """
        Initializes the ExecutionService orchestrator.

        :param sandbox_manager: Real or mock implementation of SandboxManager protocol.
        :param runner: PytestRunner instance for command construction and telemetry parsing.
        :param coverage_extractor: CoverageExtractor instance for JSON coverage parsing.
        :param source_root: Target repository root on the host (mounted at /workspace/src:ro).
                            It must be an existing directory before execution begins.
        :param staging_root: Root path for ephemeral candidate staging (mounted at /workspace/tests:ro).
        :param scratch_root: Root path for ephemeral execution outputs (mounted at /workspace/output:rw).
        :param timeout_sec: Container execution wall-clock timeout ceiling (default 30.0s, FR-13).
        :param image_tag: Hardened runner container image tag (default 'agentic-runner:0.1.0', FR-11).
        :param memory_limit: Confinement memory ceiling (default '512m', FR-14, INV-01).
        :param cpu_quota: Confinement CPU quota ceiling (default 1.0, INV-01).
        :param pids_limit: Confinement process count ceiling (default 100, INV-01).
        :param cov_source: Path inside container to measure coverage on (default '/workspace/src').
        :param environment_vars: Optional container environment variable overrides.
        :param keep_artifacts: If True, preserves staging and scratch directories for debugging.
        """
        self._sandbox_manager = sandbox_manager
        self._runner = runner or PytestRunner()
        self._coverage_extractor = coverage_extractor or CoverageExtractor()
        self._source_root = source_root.resolve() if source_root else None
        self._staging_root = staging_root.resolve() if staging_root else None
        self._scratch_root = scratch_root.resolve() if scratch_root else None
        self._timeout_sec = timeout_sec
        self._image_tag = image_tag
        self._memory_limit = memory_limit
        self._cpu_quota = cpu_quota
        self._pids_limit = pids_limit
        self._cov_source = cov_source
        self._environment_vars = dict(environment_vars or {})
        self._keep_artifacts = keep_artifacts

        self._last_summary: Optional[TestRunSummary] = None
        self._last_coverage: Optional[CoverageMetrics] = None
        self._last_evidence: Optional[ExecutionEvidence] = None

    @property
    def sandbox_manager(self) -> SandboxManager:
        """Configured SandboxManager protocol instance."""
        return self._sandbox_manager

    @property
    def runner(self) -> PytestRunner:
        """Configured PytestRunner instance."""
        return self._runner

    @property
    def coverage_extractor(self) -> CoverageExtractor:
        """Configured CoverageExtractor instance."""
        return self._coverage_extractor

    @property
    def last_summary(self) -> Optional[TestRunSummary]:
        """Parsed outcome summary from the most recent execution, if available."""
        return self._last_summary

    @property
    def last_coverage(self) -> Optional[CoverageMetrics]:
        """Parsed coverage metrics from the most recent execution, if available."""
        return self._last_coverage

    @property
    def last_evidence(self) -> Optional[ExecutionEvidence]:
        """Populated ExecutionEvidence record from the most recent execution, if available."""
        return self._last_evidence

    def _require_repository_root(self) -> Path:
        """Returns the configured target repository root or fails before sandbox setup."""
        if self._source_root is None:
            raise ExecutionError(
                "Execution requires source_root to identify the target repository root."
            )
        if not self._source_root.is_dir():
            raise ExecutionError(
                f"Execution source_root must be an existing directory: {self._source_root}"
            )
        return self._source_root

    def execute_candidate(
        self,
        plan: ExecutionPlan,
        candidate: TestCandidate,
        run_id: Optional[str] = None,
    ) -> ExecutionEvidence:
        """
        Convenience execution helper for a single TestCandidate.

        :param plan: ExecutionPlan defining the run context.
        :param candidate: Single TestCandidate entity.
        :param run_id: Optional explicit run identifier.
        :return: Populated ExecutionEvidence record.
        """
        return self.execute(
            plan=plan,
            candidates=(candidate,),
            run_id=run_id,
            candidate_id=candidate.candidate_id,
        )

    def execute(
        self,
        plan: ExecutionPlan,
        candidates: Sequence[TestCandidate] = (),
        run_id: Optional[str] = None,
        candidate_id: Optional[str] = None,
    ) -> ExecutionEvidence:
        """
        Orchestrates full sandboxed container execution and returns ExecutionEvidence.
        Stage 2 UC-05, Stage 3 Section 4.4.4.5.

        :param plan: ExecutionPlan defining target symbols and routes.
        :param candidates: Sequence of TestCandidate entities to execute.
        :param run_id: Optional explicit run identifier; defaults to candidate.run_id or plan.plan_id.
        :param candidate_id: Optional explicit candidate identifier for evidence linking.
        :return: Immutable ExecutionEvidence containing exit codes, outputs, duration, and coverage.
        :raises SourceIntegrityError: If production source files were modified (INV-02).
        :raises StagingValidationError: If candidates were provided but none passed validation.
        :raises ValueError: If neither candidates nor existing tests are available for execution.
        """
        # The target repository root is mandatory for the read-only mount and INV-02 hash check.
        repository_root = self._require_repository_root()

        # 1. Resolve effective identifiers
        effective_run_id = (
            run_id
            or (candidates[0].run_id if candidates else None)
            or plan.plan_id
        )

        # 2. Filter candidates for PASSED validation status (INV-04)
        passed_candidates = [
            c for c in candidates if c.validation_status == ValidationStatus.PASSED
        ]

        # Fail closed if candidates were passed but none were validated
        if len(candidates) > 0 and len(passed_candidates) == 0:
            raise StagingValidationError(
                "None of the provided test candidates have validation_status == PASSED. "
                "Unvalidated or rejected candidates must never be staged or executed (INV-04)."
            )

        # Fail closed if there is nothing to execute
        if len(passed_candidates) == 0 and not plan.existing_tests_to_run:
            raise ValueError(
                "No validated test candidates or existing tests available for execution."
            )

        # Resolve effective candidate ID for relational evidence linkage
        effective_candidate_id = (
            candidate_id
            or (passed_candidates[0].candidate_id if passed_candidates else "existing_tests")
        )

        # 3. Step 1 of UC-05: Source integrity pre-hash recording (INV-02)
        pre_source_hash = compute_tree_hash(repository_root)

        # 4. Set up ephemeral staging area and scratch output directory
        staging_area: Optional[CandidateStagingArea] = None
        scratch_dir: Optional[Path] = None
        created_staging_dir: Optional[Path] = None

        try:
            # Create run-scoped staging area
            session_suffix = f"{effective_run_id}_{uuid.uuid4().hex[:8]}"
            if self._staging_root is not None:
                created_staging_dir = self._staging_root / f"staging_{session_suffix}"
            else:
                created_staging_dir = Path(tempfile.mkdtemp(prefix="agentic_staging_"))

            staging_area = CandidateStagingArea(
                staging_root=created_staging_dir,
                auto_cleanup=False,
            )

            # Stage validated candidates to staging_root/tests/
            if passed_candidates:
                staging_area.stage_candidates(passed_candidates)

            # Create run-scoped scratch output directory
            if self._scratch_root is not None:
                scratch_dir = self._scratch_root / f"scratch_{session_suffix}"
                scratch_dir.mkdir(parents=True, exist_ok=True)
            else:
                scratch_dir = Path(tempfile.mkdtemp(prefix="agentic_scratch_"))

            # 5. Build volume isolation mounts (Stage 3 Section 4.4.4.5)
            read_only_mounts: Dict[Path, str] = {}
            read_only_mounts[repository_root] = "/workspace/src"

            if passed_candidates:
                read_only_mounts[staging_area.tests_dir] = "/workspace/tests"

            read_write_mounts: Dict[Path, str] = {
                scratch_dir: "/workspace/output",
            }

            # 6. Construct deterministic pytest CLI tokens via PytestRunner
            cov_report_path = "/workspace/output/coverage.json"
            if passed_candidates:
                tests_path = "/workspace/tests"
            else:
                first_test = plan.existing_tests_to_run[0]
                tests_path = f"/workspace/src/{first_test.as_posix()}"

            cmd = self._runner.construct_command(
                tests_path=tests_path,
                cov_source=self._cov_source,
                cov_report_path=cov_report_path,
            )

            # 7. Configure sandbox confinement (INV-01)
            sandbox_config = SandboxConfig(
                image_tag=self._image_tag,
                timeout_sec=self._timeout_sec,
                memory_limit=self._memory_limit,
                cpu_quota=self._cpu_quota,
                pids_limit=self._pids_limit,
                network_disabled=True,
                read_only_mounts=read_only_mounts,
                read_write_mounts=read_write_mounts,
                environment_vars=self._environment_vars,
            )

            # 8. Container lifecycle execution with guaranteed cleanup
            container_id: Optional[str] = None
            try:
                container_id = self._sandbox_manager.create_environment(sandbox_config)
                raw_result = self._sandbox_manager.execute_command(
                    container_id=container_id,
                    command=cmd,
                    workdir="/workspace",
                )
            finally:
                if container_id is not None:
                    try:
                        self._sandbox_manager.cleanup(container_id)
                    except Exception:
                        pass

            # 9. Step 9 of UC-05: Source integrity post-hash verification (INV-02)
            post_source_hash = compute_tree_hash(repository_root)
            if post_source_hash != pre_source_hash:
                raise SourceIntegrityError(
                    f"Safety Invariant INV-02 violated: Production source files under "
                    f"'{repository_root}' were modified during test execution! "
                    f"Pre-execution SHA-256: {pre_source_hash}, "
                    f"Post-execution SHA-256: {post_source_hash}. "
                    f"Halting immediately (UC-05 Extension 9a)."
                )

            # 10. Parse test outcomes & captured telemetry
            summary = self._runner.parse_test_outcomes(
                stdout=raw_result.stdout,
                stderr=raw_result.stderr,
                exit_code=raw_result.exit_code,
            )

            # 11. Parse coverage telemetry from generated coverage.json
            coverage_file = scratch_dir / "coverage.json"
            coverage_metrics: CoverageMetrics
            if coverage_file.is_file():
                coverage_metrics = self._coverage_extractor.parse_coverage_json(
                    coverage_file,
                    strict=False,
                )
            else:
                coverage_metrics = CoverageMetrics(
                    line_coverage=0.0,
                    branch_coverage=0.0,
                    is_valid=False,
                    error_message=f"Coverage report file '{coverage_file}' not found.",
                )

            # 12. Package immutable ExecutionEvidence entity
            evidence_id = f"ev-{uuid.uuid4().hex[:12]}"
            evidence = ExecutionEvidence(
                evidence_id=evidence_id,
                run_id=effective_run_id,
                candidate_id=effective_candidate_id,
                exit_code=raw_result.exit_code,
                stdout=raw_result.stdout,
                stderr=raw_result.stderr,
                duration_sec=raw_result.duration_sec,
                timed_out=summary.timed_out,
                line_coverage=coverage_metrics.line_coverage,
                branch_coverage=coverage_metrics.branch_coverage,
                traceback=summary.traceback,
                created_at=datetime.now(timezone.utc),
            )

            # Cache detailed telemetry for caller introspection
            self._last_summary = summary
            self._last_coverage = coverage_metrics
            self._last_evidence = evidence

            return evidence

        finally:
            # 13. Fail-closed ephemeral mount teardown
            if not self._keep_artifacts:
                if staging_area is not None:
                    try:
                        staging_area.cleanup()
                    except Exception:
                        pass
                elif created_staging_dir is not None and created_staging_dir.exists():
                    shutil.rmtree(created_staging_dir, ignore_errors=True)

                if scratch_dir is not None and scratch_dir.exists():
                    shutil.rmtree(scratch_dir, ignore_errors=True)
