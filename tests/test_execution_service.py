"""
Offline unit and integration tests for ExecutionService.
Stage 2 Section 4.3.2.5 (UC-05), Stage 3 Section 4.4.4.5.
Iteration 1.5 Slice D.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import pytest

from agentic_test.core.models import (
    ExecutionEvidence,
    ExecutionPlan,
    SymbolContract,
    SymbolType,
    TestCandidate,
    ValidationStatus,
    WorkflowRoute,
)
from agentic_test.core.protocols.sandbox import (
    ExecutionRawResult,
    SandboxConfig,
    SandboxExecutionError,
    SandboxInitializationError,
)
from agentic_test.execution.mock_sandbox import MockSandboxManager
from agentic_test.execution.runner import PytestRunner
from agentic_test.execution.service import (
    ExecutionService,
    ProductionSourceIntegrityViolation,
    SourceIntegrityError,
    compute_tree_hash,
)
from agentic_test.execution.staging import StagingValidationError


class SimulatedContainerSandboxManager(MockSandboxManager):
    """
    Subclass of MockSandboxManager that simulates container side-effects,
    such as writing coverage.json to the mounted /workspace/output directory,
    simulating container command execution errors,
    or simulating unauthorized file modifications to test INV-02.
    """

    def __init__(
        self,
        coverage_payload: Optional[str] = None,
        coverage_payloads: Optional[Sequence[Optional[str]]] = None,
        tamper_target: Optional[Path] = None,
        default_result: Optional[ExecutionRawResult] = None,
        execute_error: Optional[Exception] = None,
    ) -> None:
        super().__init__(default_result=default_result)
        self.coverage_payload = coverage_payload
        self.coverage_payloads = list(coverage_payloads or ())
        self.tamper_target = tamper_target
        self.execute_error = execute_error

    def execute_command(
        self,
        container_id: str,
        command: List[str],
        workdir: str = "/workspace",
    ) -> ExecutionRawResult:
        config = self._containers[container_id]

        # Simulate writing the coverage artifact requested by the pytest command.
        coverage_payload = (
            self.coverage_payloads.pop(0)
            if self.coverage_payloads
            else self.coverage_payload
        )
        if coverage_payload is not None:
            report_option = next(
                (token for token in command if token.startswith("--cov-report=json:")),
                "--cov-report=json:/workspace/output/coverage.json",
            )
            report_filename = Path(report_option.split(":", 1)[1]).name
            for host_path, container_path in config.read_write_mounts.items():
                if container_path == "/workspace/output":
                    host_path.mkdir(parents=True, exist_ok=True)
                    (host_path / report_filename).write_text(
                        coverage_payload, encoding="utf-8"
                    )

        # Simulate host tampering to verify INV-02 violation detection
        if self.tamper_target is not None and self.tamper_target.exists():
            if self.tamper_target.is_file():
                self.tamper_target.write_text("TAMPERED_CONTENT\n", encoding="utf-8")
            elif self.tamper_target.is_dir():
                (self.tamper_target / "malicious.py").write_text("# injected\n", encoding="utf-8")

        if self.execute_error is not None:
            raise self.execute_error

        return super().execute_command(container_id, command, workdir)


def _make_candidate(
    candidate_id: str = "cand-001",
    run_id: str = "run-001",
    target_symbol: str = "calculator.add",
    test_path: Path = Path("tests/test_calc.py"),
    code: str = "def test_add(): assert add(1, 2) == 3\n",
    status: ValidationStatus = ValidationStatus.PASSED,
) -> TestCandidate:
    return TestCandidate(
        candidate_id=candidate_id,
        run_id=run_id,
        target_symbol_name=target_symbol,
        test_file_path=test_path,
        candidate_code=code,
        imports=("from app.calc import add",),
        validation_status=status,
    )


def _make_plan(
    plan_id: str = "plan-001",
    route: WorkflowRoute = WorkflowRoute.ROUTE_TO_TEST_GENERATION,
    existing_tests: Tuple[Path, ...] = (),
) -> ExecutionPlan:
    return ExecutionPlan(
        plan_id=plan_id,
        route=route,
        target_symbols=(),
        existing_tests_to_run=existing_tests,
        rationale="Unit test generation execution",
        decision_hash="0" * 64,
    )


def _make_repository_root(tmp_path: Path) -> Path:
    """Creates an empty target repository root for execution-service tests."""
    repository_root = tmp_path / "repository"
    repository_root.mkdir(exist_ok=True)
    return repository_root


def test_execution_service_requires_repository_root_argument() -> None:
    """The repository root is a construction-time requirement for execution safety."""
    with pytest.raises(TypeError, match="missing 1 required positional argument: 'source_root'"):
        ExecutionService(sandbox_manager=SimulatedContainerSandboxManager())


VALID_COVERAGE_JSON = json.dumps({
    "meta": {"version": "7.4.4"},
    "files": {
        "src/calc.py": {
            "executed_lines": [1, 2, 4, 5],
            "missing_lines": [7],
            "summary": {
                "covered_lines": 4,
                "num_statements": 5,
                "percent_covered": 80.0,
                "missing_lines": 1,
                "excluded_lines": 0,
            },
        }
    },
    "totals": {
        "covered_lines": 4,
        "num_statements": 5,
        "percent_covered": 80.0,
        "missing_lines": 1,
        "excluded_lines": 0,
        "covered_branches": 3,
        "num_branches": 4,
        "missing_branches": 1,
    },
})


BASELINE_COVERAGE_JSON = json.dumps({
    "totals": {
        "covered_lines": 3,
        "num_statements": 5,
        "percent_covered": 60.0,
        "missing_lines": 2,
        "excluded_lines": 0,
        "covered_branches": 2,
        "num_branches": 4,
        "missing_branches": 2,
    },
})


# =====================================================================
# 1. Successful Execution & Telemetry Extraction Tests
# =====================================================================

def test_successful_candidate_execution(tmp_path: Path) -> None:
    """
    Verifies full execution lifecycle for a passing test candidate:
    - Command construction & container configuration.
    - Successful pytest outcome parsing (2 passed).
    - Coverage extraction from coverage.json.
    - Evidence assembly and property availability.
    - Ephemeral cleanup of staging and container.
    """
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "calc.py").write_text("def add(a, b): return a + b\n", encoding="utf-8")

    sandbox = SimulatedContainerSandboxManager(
        coverage_payload=VALID_COVERAGE_JSON,
    )
    sandbox.script_success(
        stdout="=== 2 passed in 0.42s ===\n",
        duration_sec=0.42,
    )

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=source_dir,
    )

    plan = _make_plan()
    candidate = _make_candidate()

    evidence = service.execute(plan, [candidate])

    # Assert ExecutionEvidence values
    assert evidence.exit_code == 0
    assert evidence.duration_sec == 0.42
    assert evidence.timed_out is False
    assert evidence.line_coverage == 80.0
    assert evidence.branch_coverage == 75.0
    assert evidence.line_coverage_delta is None
    assert evidence.branch_coverage_delta is None
    assert evidence.traceback is None
    assert evidence.candidate_id == candidate.candidate_id
    assert evidence.run_id == candidate.run_id
    assert evidence.evidence_id.startswith("ev-")

    # Assert last summary & coverage properties
    assert service.last_summary is not None
    assert service.last_summary.passed == 2
    assert service.last_summary.success is True

    assert service.last_coverage is not None
    assert service.last_coverage.is_valid is True
    assert service.last_coverage.covered_lines == 4

    assert service.last_evidence == evidence

    # Assert container cleanup (INV-01)
    assert sandbox.active_containers == []
    assert len(sandbox.terminated_containers) == 1

    # Assert command execution was recorded
    assert sandbox.call_count == 1
    call = sandbox.calls[0]
    assert call["command"] == [
        "pytest",
        "/workspace/tests",
        "--cov=/workspace/src",
        "--cov-branch",
        "--cov-report=json:/workspace/output/coverage.json",
        "-o",
        "cache_dir=/tmp/.pytest_cache",
    ]


def test_candidate_execution_records_deltas_against_regression_baseline(
    tmp_path: Path,
) -> None:
    """Generated-test coverage is compared with the selected regression baseline."""
    sandbox = SimulatedContainerSandboxManager(
        coverage_payloads=(BASELINE_COVERAGE_JSON, VALID_COVERAGE_JSON),
    )
    sandbox.script_success(stdout="=== 1 passed in 0.1s ===\n")
    sandbox.script_success(stdout="=== 3 passed in 0.2s ===\n")
    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan(existing_tests=(Path("tests/test_existing.py"),))

    evidence = service.execute(plan, candidates=(_make_candidate(),))

    assert evidence.line_coverage == 80.0
    assert evidence.branch_coverage == 75.0
    assert evidence.line_coverage_delta == 20.0
    assert evidence.branch_coverage_delta == 25.0
    assert sandbox.call_count == 2
    assert sandbox.calls[0]["command"][1] == "/workspace/src/tests/test_existing.py"
    assert "--cov-report=json:/workspace/output/coverage-baseline.json" in sandbox.calls[0]["command"]
    assert sandbox.calls[1]["command"][:3] == [
        "pytest",
        "/workspace/tests",
        "/workspace/src/tests/test_existing.py",
    ]
    assert "--cov-report=json:/workspace/output/coverage.json" in sandbox.calls[1]["command"]
    assert len(sandbox.terminated_containers) == 2


def test_failing_candidate_captures_traceback(tmp_path: Path) -> None:
    """
    Verifies that test assertion failure (exit code 1) captures the diagnostic
    traceback into ExecutionEvidence without raising an unhandled exception.
    """
    sandbox = SimulatedContainerSandboxManager()
    sandbox.script_failure(
        exit_code=1,
        stdout=(
            "=== FAILURES ===\n"
            "___ test_add ___\n"
            "    def test_add():\n"
            ">       assert add(1, 2) == 4\n"
            "E       assert 3 == 4\n"
            "tests/test_calc.py:3: AssertionError\n"
            "=== 1 failed in 0.15s ===\n"
        ),
        duration_sec=0.15,
    )

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan()
    candidate = _make_candidate()

    evidence = service.execute(plan, [candidate])

    assert evidence.exit_code == 1
    assert evidence.timed_out is False
    assert evidence.traceback is not None
    assert "AssertionError" in evidence.traceback
    assert "assert 3 == 4" in evidence.traceback
    assert service.last_summary is not None
    assert service.last_summary.failed == 1
    assert service.last_summary.success is False


def test_collection_error_captured(tmp_path: Path) -> None:
    """
    Verifies that pytest collection errors (exit code 2) are captured cleanly.
    """
    sandbox = SimulatedContainerSandboxManager()
    sandbox.script_failure(
        exit_code=2,
        stdout="=== 1 error in 0.08s ===\n",
        stderr="ModuleNotFoundError: No module named 'nonexistent_package'\n",
        duration_sec=0.08,
    )

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan()
    candidate = _make_candidate()

    evidence = service.execute(plan, [candidate])

    assert evidence.exit_code == 2
    assert evidence.traceback is not None
    assert "ModuleNotFoundError" in evidence.traceback
    assert service.last_summary is not None
    assert service.last_summary.errors == 1


def test_no_tests_collected_exit_code_5(tmp_path: Path) -> None:
    """
    Verifies that exit code 5 (no tests collected) is recorded properly in evidence.
    """
    sandbox = SimulatedContainerSandboxManager()
    sandbox.script_failure(
        exit_code=5,
        stdout="=== no tests ran in 0.02s ===\n",
        duration_sec=0.02,
    )

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan()
    candidate = _make_candidate()

    evidence = service.execute(plan, [candidate])

    assert evidence.exit_code == 5
    assert evidence.timed_out is False
    assert service.last_summary is not None
    assert service.last_summary.exit_code == 5


# =====================================================================
# 2. Timeout & Resource Limits Confinement Tests (INV-01)
# =====================================================================

def test_execution_timeout_inv01(tmp_path: Path) -> None:
    """
    Verifies that container timeout (exit code 124) sets timed_out=True
    on the resulting ExecutionEvidence entity (UC-05 Extension 6a).
    """
    sandbox = SimulatedContainerSandboxManager()
    sandbox.script_timeout(duration_sec=30.0)

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
        timeout_sec=30.0,
    )
    plan = _make_plan()
    candidate = _make_candidate()

    evidence = service.execute(plan, [candidate])

    assert evidence.exit_code == 124
    assert evidence.timed_out is True
    assert evidence.duration_sec == 30.0
    assert service.last_summary is not None
    assert service.last_summary.timed_out is True


def test_execution_oom_killed_inv01(tmp_path: Path) -> None:
    """
    Verifies that container memory exhaustion (exit code 137) is parsed as OOM
    (UC-05 Extension 6b).
    """
    sandbox = SimulatedContainerSandboxManager()
    sandbox.script_oom(duration_sec=0.5)

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan()
    candidate = _make_candidate()

    evidence = service.execute(plan, [candidate])

    assert evidence.exit_code == 137
    assert service.last_summary is not None
    assert service.last_summary.oom_killed is True
    assert evidence.traceback is not None
    assert "out of memory" in evidence.traceback.lower()


def test_sandbox_confinement_config_parameters(tmp_path: Path) -> None:
    """
    Verifies that ExecutionService configures SandboxConfig strictly with
    INV-01 security invariants (network_disabled=True, limits, non-root).
    """
    source_dir = tmp_path / "src"
    source_dir.mkdir()

    sandbox = SimulatedContainerSandboxManager()
    sandbox.script_success()

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=source_dir,
        timeout_sec=45.0,
        memory_limit="256m",
        cpu_quota=2.0,
        pids_limit=50,
        image_tag="agentic-runner:custom",
    )

    plan = _make_plan()
    candidate = _make_candidate()
    service.execute(plan, [candidate])

    assert len(sandbox.created_configs) == 1
    config = list(sandbox.created_configs.values())[0]

    assert config.network_disabled is True
    assert config.timeout_sec == 45.0
    assert config.memory_limit == "256m"
    assert config.cpu_quota == 2.0
    assert config.pids_limit == 50
    assert config.image_tag == "agentic-runner:custom"
    assert config.read_only_mounts[source_dir.resolve()] == "/workspace/src"


# =====================================================================
# 3. Source Tree Integrity Verification Tests (INV-02)
# =====================================================================

def test_source_integrity_preservation_inv02(tmp_path: Path) -> None:
    """
    Verifies that when source files are untouched, SHA-256 pre/post verification passes cleanly.
    """
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    (source_dir / "mod.py").write_text("print('hello')\n", encoding="utf-8")

    pre_hash = compute_tree_hash(source_dir)

    sandbox = SimulatedContainerSandboxManager()
    sandbox.script_success()

    service = ExecutionService(sandbox_manager=sandbox, source_root=source_dir)
    plan = _make_plan()
    candidate = _make_candidate()

    evidence = service.execute(plan, [candidate])
    assert evidence.exit_code == 0
    assert compute_tree_hash(source_dir) == pre_hash


def test_source_integrity_violation_raises_error_inv02(tmp_path: Path) -> None:
    """
    Verifies that tampering with source files raises SourceIntegrityError
    and ProductionSourceIntegrityViolation (UC-05 Extension 9a).
    """
    source_dir = tmp_path / "src"
    source_dir.mkdir()
    target_file = source_dir / "mod.py"
    target_file.write_text("def pristine(): pass\n", encoding="utf-8")

    # Simulate tampering during container execution
    sandbox = SimulatedContainerSandboxManager(tamper_target=target_file)
    sandbox.script_success()

    service = ExecutionService(sandbox_manager=sandbox, source_root=source_dir)
    plan = _make_plan()
    candidate = _make_candidate()

    with pytest.raises(SourceIntegrityError) as exc_info:
        service.execute(plan, [candidate])

    assert issubclass(ProductionSourceIntegrityViolation, SourceIntegrityError)
    assert "INV-02 violated" in str(exc_info.value)
    assert "UC-05 Extension 9a" in str(exc_info.value)

    # Verify container was still cleaned up despite integrity error
    assert sandbox.active_containers == []
    assert len(sandbox.terminated_containers) == 1


def test_source_integrity_violation_raises_even_when_sandbox_execution_fails(
    tmp_path: Path,
) -> None:
    """INV-02 remains enforced when the sandbox raises before returning telemetry."""
    source_dir = tmp_path / "repository"
    source_dir.mkdir()
    target_file = source_dir / "calc.py"
    target_file.write_text("def pristine(): pass\n", encoding="utf-8")
    sandbox = SimulatedContainerSandboxManager(
        tamper_target=target_file,
        execute_error=SandboxExecutionError("Simulated container crash"),
    )
    service = ExecutionService(sandbox_manager=sandbox, source_root=source_dir)

    with pytest.raises(SourceIntegrityError, match="INV-02 violated"):
        service.execute(_make_plan(), candidates=(_make_candidate(),))

    assert sandbox.active_containers == []


# =====================================================================
# 4. Coverage Extraction & Edge Cases
# =====================================================================

def test_missing_coverage_json_handled_safely(tmp_path: Path) -> None:
    """
    Verifies that when coverage.json is absent, line_coverage defaults to 0.0
    and is_valid is marked False without crashing.
    """
    sandbox = SimulatedContainerSandboxManager(coverage_payload=None)
    sandbox.script_success()

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan()
    candidate = _make_candidate()

    evidence = service.execute(plan, [candidate])

    assert evidence.line_coverage == 0.0
    assert evidence.branch_coverage == 0.0
    assert service.last_coverage is not None
    assert service.last_coverage.is_valid is False
    assert "not found" in (service.last_coverage.error_message or "").lower()


def test_malformed_coverage_json_handled_safely(tmp_path: Path) -> None:
    """
    Verifies that malformed coverage JSON falls back gracefully without crashing.
    """
    sandbox = SimulatedContainerSandboxManager(coverage_payload="{not-valid-json}")
    sandbox.script_success()

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan()
    candidate = _make_candidate()

    evidence = service.execute(plan, [candidate])

    assert evidence.line_coverage == 0.0
    assert service.last_coverage is not None
    assert service.last_coverage.is_valid is False
    assert "malformed" in (service.last_coverage.error_message or "").lower()


# =====================================================================
# 5. Fail-Closed Cleanup & Error Guarantees
# =====================================================================

def test_container_and_mounts_cleaned_up_on_sandbox_execution_error(tmp_path: Path) -> None:
    """
    Verifies that if SandboxManager.execute_command raises an exception,
    the container is terminated and ephemeral staging directories are removed.
    """
    staging_root = tmp_path / "staging_parent"
    staging_root.mkdir()
    scratch_root = tmp_path / "scratch_parent"
    scratch_root.mkdir()

    sandbox = SimulatedContainerSandboxManager(
        execute_error=SandboxExecutionError("Simulated container crash")
    )

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
        staging_root=staging_root,
        scratch_root=scratch_root,
    )
    plan = _make_plan()
    candidate = _make_candidate()

    with pytest.raises(SandboxExecutionError):
        service.execute(plan, [candidate])

    # Container terminated
    assert sandbox.active_containers == []
    assert len(sandbox.terminated_containers) == 1

    # Staging and scratch directories under roots removed
    assert list(staging_root.iterdir()) == []
    assert list(scratch_root.iterdir()) == []


def test_mounts_cleaned_up_on_sandbox_initialization_error(tmp_path: Path) -> None:
    """
    Verifies that if SandboxManager.create_environment fails (SandboxInitializationError),
    ephemeral staging and scratch directories are still reliably cleaned up.
    """
    staging_root = tmp_path / "staging_parent"
    staging_root.mkdir()
    scratch_root = tmp_path / "scratch_parent"
    scratch_root.mkdir()

    sandbox = SimulatedContainerSandboxManager()
    sandbox.enqueue_error(SandboxInitializationError("Simulated daemon failure"))

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
        staging_root=staging_root,
        scratch_root=scratch_root,
    )
    plan = _make_plan()
    candidate = _make_candidate()

    with pytest.raises(SandboxInitializationError):
        service.execute(plan, [candidate])

    # No active containers
    assert sandbox.active_containers == []

    # Staging and scratch directories under roots removed
    assert list(staging_root.iterdir()) == []
    assert list(scratch_root.iterdir()) == []


def test_unvalidated_candidates_rejected_inv04(tmp_path: Path) -> None:
    """
    Verifies that candidates without validation_status == PASSED are rejected
    before staging and never executed (INV-04).
    """
    sandbox = SimulatedContainerSandboxManager()
    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )

    plan = _make_plan()
    bad_cand = _make_candidate(status=ValidationStatus.REJECTED_SECURITY)

    with pytest.raises(StagingValidationError) as exc_info:
        service.execute(plan, [bad_cand])

    assert "PASSED" in str(exc_info.value)
    assert sandbox.call_count == 0
    assert sandbox.active_containers == []


def test_empty_candidates_and_no_existing_tests_rejected(tmp_path: Path) -> None:
    """
    Verifies that an error is raised when neither candidates nor existing tests are provided.
    """
    sandbox = SimulatedContainerSandboxManager()
    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan(existing_tests=())

    with pytest.raises(ValueError) as exc_info:
        service.execute(plan, candidates=())

    assert "No validated test candidates or existing tests" in str(exc_info.value)
    assert sandbox.call_count == 0


def test_existing_tests_execution_without_candidates(tmp_path: Path) -> None:
    """
    Verifies execution of existing repository tests when candidates sequence is empty.
    """
    sandbox = SimulatedContainerSandboxManager()
    sandbox.script_success(stdout="=== 5 passed in 0.8s ===\n")

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan(
        plan_id="plan-regression-01",
        existing_tests=(
            Path("tests/test_suite.py"),
            Path("tests/test_regression.py"),
        ),
    )

    evidence = service.execute(plan, candidates=())

    assert evidence.exit_code == 0
    assert evidence.candidate_id == "existing_tests"
    assert evidence.run_id == "plan-regression-01"

    assert sandbox.call_count == 1
    cmd = sandbox.calls[0]["command"]
    assert cmd[0] == "pytest"
    assert cmd[1:3] == [
        "/workspace/src/tests/test_suite.py",
        "/workspace/src/tests/test_regression.py",
    ]


def test_candidate_execution_includes_planned_regression_tests(tmp_path: Path) -> None:
    """Generated candidates and planner-selected regressions share one pytest session."""
    sandbox = SimulatedContainerSandboxManager()
    sandbox.script_success(stdout="=== 3 passed in 0.2s ===\n")
    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan(
        existing_tests=(
            Path("tests/test_existing_one.py"),
            Path("tests/test_existing_two.py"),
        ),
    )

    evidence = service.execute(plan, candidates=(_make_candidate(),))

    assert evidence.exit_code == 0
    assert sandbox.call_count == 2
    assert sandbox.calls[0]["command"][1:3] == [
        "/workspace/src/tests/test_existing_one.py",
        "/workspace/src/tests/test_existing_two.py",
    ]
    assert sandbox.calls[1]["command"][:4] == [
        "pytest",
        "/workspace/tests",
        "/workspace/src/tests/test_existing_one.py",
        "/workspace/src/tests/test_existing_two.py",
    ]


def test_execute_candidate_convenience_method(tmp_path: Path) -> None:
    """
    Verifies the execute_candidate convenience helper.
    """
    sandbox = SimulatedContainerSandboxManager()
    sandbox.script_success(stdout="=== 1 passed in 0.1s ===\n")

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
    )
    plan = _make_plan()
    cand = _make_candidate(candidate_id="cand-singular-42")

    evidence = service.execute_candidate(plan, cand)

    assert evidence.exit_code == 0
    assert evidence.candidate_id == "cand-singular-42"


def test_keep_artifacts_preserves_directories(tmp_path: Path) -> None:
    """
    Verifies that keep_artifacts=True preserves staging and scratch directories for inspection.
    """
    staging_root = tmp_path / "staging_parent"
    staging_root.mkdir()
    scratch_root = tmp_path / "scratch_parent"
    scratch_root.mkdir()

    sandbox = SimulatedContainerSandboxManager(coverage_payload=VALID_COVERAGE_JSON)
    sandbox.script_success()

    service = ExecutionService(
        sandbox_manager=sandbox,
        source_root=_make_repository_root(tmp_path),
        staging_root=staging_root,
        scratch_root=scratch_root,
        keep_artifacts=True,
    )
    plan = _make_plan()
    cand = _make_candidate()

    evidence = service.execute(plan, [cand])
    assert evidence.exit_code == 0

    # Verify directories still exist
    assert len(list(staging_root.iterdir())) == 1
    assert len(list(scratch_root.iterdir())) == 1


# =====================================================================
# 6. Tree Hash Determinism Tests
# =====================================================================

def test_compute_tree_hash_determinism(tmp_path: Path) -> None:
    """
    Verifies that compute_tree_hash produces deterministic SHA-256 hashes
    and ignores caches and temporary artifacts.
    """
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    (root / "b.py").write_text("b = 2\n", encoding="utf-8")

    hash1 = compute_tree_hash(root)
    hash2 = compute_tree_hash(root)
    assert hash1 == hash2
    assert len(hash1) == 64

    # Caches must be ignored
    cache_dir = root / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "a.cpython-311.pyc").write_bytes(b"\x00\x01\x02")

    assert compute_tree_hash(root) == hash1

    # Modifying a tracked file must change the hash
    (root / "a.py").write_text("a = 999\n", encoding="utf-8")
    assert compute_tree_hash(root) != hash1
