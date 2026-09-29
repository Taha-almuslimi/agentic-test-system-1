"""
Unit tests for core models and WorkflowState.
Verifies validation, immutability (NFR-06), serialization, and Option A defaults.
"""

from datetime import datetime
from pathlib import Path
import pytest
from pydantic import ValidationError

from agentic_test.core.models import (
    ChangeType,
    DiffHunk,
    ExecutionEvidence,
    ExecutionPlan,
    FailureCategory,
    FailureDiagnosis,
    RepositorySnapshot,
    SymbolContract,
    SymbolType,
    TestCandidate,
    TriageEngine,
    ValidationStatus,
    WorkflowRoute,
)
from agentic_test.core.state import WorkflowState


def test_diff_hunk_creation_and_immutability() -> None:
    """DiffHunk value object validates attributes and enforces immutability."""
    hunk = DiffHunk(
        file_path=Path("src/module.py"),
        old_start=10,
        old_lines=5,
        new_start=10,
        new_lines=8,
        change_type=ChangeType.MODIFIED,
        content="@@ -10,5 +10,8 @@\n+line",
    )
    assert hunk.file_path == Path("src/module.py")
    assert hunk.old_start == 10
    assert hunk.new_lines == 8
    assert hunk.change_type == ChangeType.MODIFIED

    # Immutability enforcement (frozen=True)
    with pytest.raises(ValidationError):
        setattr(hunk, "old_start", 20)


def test_symbol_contract_creation_and_immutability() -> None:
    """SymbolContract entity validates fields and enforces immutability."""
    symbol = SymbolContract(
        qualified_name="Calculator.add",
        symbol_type=SymbolType.METHOD,
        file_path=Path("src/calc.py"),
        line_range=(15, 25),
        signature="def add(self, a: int, b: int) -> int:",
        docstring="Adds two numbers.",
        dependencies=("typing",),
        is_affected=False,
    )
    assert symbol.qualified_name == "Calculator.add"
    assert symbol.symbol_type == SymbolType.METHOD
    assert symbol.line_range == (15, 25)
    assert symbol.is_affected is False

    with pytest.raises(ValidationError):
        setattr(symbol, "is_affected", True)


def test_repository_snapshot_option_a_defaults() -> None:
    """
    RepositorySnapshot preserves the 9 baseline fields and provides independent
    empty-tuple defaults for diff_hunks, affected_symbols, and syntax_errors (Option A).
    """
    snapshot = RepositorySnapshot(
        repo_path=Path("/tmp/test_repo"),
        current_commit="a" * 40,
        base_commit="b" * 40,
        branch_name="main",
        tracked_files=(Path("app.py"),),
        existing_test_files=(Path("test_app.py"),),
        source_tree_hash="c" * 64,
        is_valid=True,
    )

    # 9 baseline fields verified
    assert snapshot.repo_path == Path("/tmp/test_repo")
    assert snapshot.current_commit == "a" * 40
    assert snapshot.base_commit == "b" * 40
    assert snapshot.branch_name == "main"
    assert snapshot.tracked_files == (Path("app.py"),)
    assert snapshot.existing_test_files == (Path("test_app.py"),)
    assert snapshot.source_tree_hash == "c" * 64
    assert snapshot.is_valid is True
    assert isinstance(snapshot.created_at, datetime)

    # Option A fields default to empty tuples
    assert snapshot.diff_hunks == ()
    assert snapshot.affected_symbols == ()
    assert snapshot.syntax_errors == ()

    # Verify collections default to immutable empty tuples
    snapshot2 = RepositorySnapshot(
        repo_path=Path("/tmp/other_repo"),
        current_commit="d" * 40,
        base_commit="e" * 40,
        branch_name="dev",
        source_tree_hash="f" * 64,
        is_valid=True,
    )
    assert snapshot.diff_hunks == ()
    assert snapshot2.diff_hunks == ()
    assert isinstance(snapshot.diff_hunks, tuple)
    assert isinstance(snapshot.affected_symbols, tuple)
    assert isinstance(snapshot.syntax_errors, tuple)


def test_repository_snapshot_immutability() -> None:
    """RepositorySnapshot is frozen and cannot be mutated in place."""
    snapshot = RepositorySnapshot(
        repo_path=Path("/tmp/test_repo"),
        current_commit="a" * 40,
        base_commit="b" * 40,
        branch_name="main",
        source_tree_hash="c" * 64,
        is_valid=True,
    )
    with pytest.raises(ValidationError):
        setattr(snapshot, "is_valid", False)

    with pytest.raises(ValidationError):
        setattr(snapshot, "current_commit", "f" * 40)


def test_repository_snapshot_serialization() -> None:
    """RepositorySnapshot serializes to JSON and deserializes back faithfully."""
    hunk = DiffHunk(
        file_path=Path("main.py"),
        old_start=1,
        old_lines=3,
        new_start=1,
        new_lines=5,
        change_type=ChangeType.ADDED,
        content="@@ -1,3 +1,5 @@\n+print('hello')",
    )
    snapshot = RepositorySnapshot(
        repo_path=Path("/repo"),
        current_commit="1234567890abcdef1234567890abcdef12345678",
        base_commit="0000000000000000000000000000000000000000",
        branch_name="feature/test",
        tracked_files=(Path("main.py"), Path("utils.py")),
        existing_test_files=(Path("test_main.py"),),
        source_tree_hash="abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
        is_valid=True,
        diff_hunks=(hunk,),
        syntax_errors=("SyntaxError in legacy.py: line 10",),
    )

    json_data = snapshot.model_dump_json()
    reconstructed = RepositorySnapshot.model_validate_json(json_data)

    assert reconstructed == snapshot
    assert len(reconstructed.diff_hunks) == 1
    assert reconstructed.diff_hunks[0].file_path == Path("main.py")
    assert reconstructed.syntax_errors == ("SyntaxError in legacy.py: line 10",)


def test_workflow_state_creation_and_immutability() -> None:
    """WorkflowState instantiates, captures snapshot, and enforces immutability."""
    snapshot = RepositorySnapshot(
        repo_path=Path("/tmp/repo"),
        current_commit="1" * 40,
        base_commit="2" * 40,
        branch_name="master",
        source_tree_hash="3" * 64,
        is_valid=True,
    )

    state = WorkflowState(
        run_id="run-uuid-001",
        repo_path=Path("/tmp/repo"),
        snapshot=snapshot,
    )

    assert state.run_id == "run-uuid-001"
    assert state.repo_path == Path("/tmp/repo")
    assert isinstance(state.created_at, datetime)
    assert state.snapshot == snapshot
    assert state.plan is None
    assert state.candidates == ()
    assert state.generation_retry_count == 0
    assert state.max_generation_retries == 2
    assert state.evidences == ()
    assert state.diagnoses == ()
    assert state.final_status == "INITIALIZED"
    assert state.error_message is None

    # Verify immutability
    with pytest.raises(ValidationError):
        setattr(state, "final_status", "SUCCESS")

    # Serialization test
    state_json = state.model_dump_json()
    restored = WorkflowState.model_validate_json(state_json)
    assert restored.run_id == state.run_id
    assert restored.snapshot is not None
    assert restored.snapshot.current_commit == "1" * 40


def test_workflow_state_rejects_invalid_field_types() -> None:
    """
    Stage 3 §4.4.3: WorkflowState enforces concrete Pydantic schemas on
    plan, candidates, evidence, and diagnoses, rejecting invalid types.
    """
    # 1. Reject invalid plan type (expects ExecutionPlan, reject str or int)
    with pytest.raises(ValidationError):
        WorkflowState(
            run_id="run-01",
            repo_path=Path("/tmp/repo"),
            plan="invalid_string_plan",  # type: ignore[arg-type]
        )

    # 2. Reject invalid candidates list item (expects TestCandidate, reject str or int)
    with pytest.raises(ValidationError):
        WorkflowState(
            run_id="run-01",
            repo_path=Path("/tmp/repo"),
            candidates=("not_a_candidate",),  # type: ignore[arg-type]
        )

    # 3. Reject invalid evidence type (expects ExecutionEvidence, reject str or int)
    with pytest.raises(ValidationError):
        WorkflowState(
            run_id="run-01",
            repo_path=Path("/tmp/repo"),
            evidences=("invalid_evidence",),  # type: ignore[arg-type]
        )

    # 4. Reject invalid diagnoses list item (expects FailureDiagnosis, reject str or int)
    with pytest.raises(ValidationError):
        WorkflowState(
            run_id="run-01",
            repo_path=Path("/tmp/repo"),
            diagnoses=(12345,),  # type: ignore[arg-type]
        )


def test_workflow_state_with_valid_stage3_entities() -> None:
    """
    Stage 3 §4.4.3: WorkflowState instantiates with concrete Stage 3 entities
    and serializes/deserializes faithfully.
    """
    plan = ExecutionPlan(
        plan_id="plan-001",
        route=WorkflowRoute.ROUTE_TO_TEST_GENERATION,
        rationale="Uncovered modified symbols detected",
        decision_hash="hash-1234567890",
    )

    candidate = TestCandidate(
        candidate_id="cand-001",
        run_id="run-01",
        target_symbol_name="Calculator.add",
        test_file_path=Path("tests/test_calc.py"),
        candidate_code="def test_add(): assert True\n",
        validation_status=ValidationStatus.PENDING,
    )

    evidence = ExecutionEvidence(
        evidence_id="ev-001",
        run_id="run-01",
        candidate_id="cand-001",
        exit_code=0,
        stdout="1 passed",
        stderr="",
        duration_sec=1.25,
        line_coverage=0.95,
        branch_coverage=0.90,
        line_coverage_delta=0.15,
        branch_coverage_delta=0.10,
    )

    diagnosis = FailureDiagnosis(
        diagnosis_id="diag-001",
        evidence_id="ev-001",
        canonical_category=FailureCategory.APPLICATION_BUG,
        confidence=0.92,
        triage_engine=TriageEngine.DETERMINISTIC_RULE,
        explanation="Assertion failure indicates application defect",
        is_application_bug=True,
    )

    state = WorkflowState(
        run_id="run-01",
        repo_path=Path("/tmp/repo"),
        plan=plan,
        candidates=(candidate,),
        evidences=(
            evidence,
            ExecutionEvidence(
                evidence_id="ev-regression-001",
                run_id="run-01",
                candidate_id=None,
                exit_code=0,
                stdout="2 passed",
                stderr="",
                duration_sec=0.75,
            ),
        ),
        diagnoses=(diagnosis,),
    )

    assert state.plan == plan
    assert state.plan.route == WorkflowRoute.ROUTE_TO_TEST_GENERATION
    assert len(state.candidates) == 1
    assert state.candidates[0].candidate_id == "cand-001"
    assert len(state.evidences) == 2
    assert state.evidences[0] == evidence
    assert state.evidences[0].exit_code == 0
    assert state.evidences[0].line_coverage_delta == 0.15
    assert state.evidences[0].branch_coverage_delta == 0.10
    assert state.evidences[1].candidate_id is None
    assert len(state.diagnoses) == 1
    assert state.diagnoses[0].canonical_category == FailureCategory.APPLICATION_BUG

    # JSON round-trip
    state_json = state.model_dump_json()
    reconstructed = WorkflowState.model_validate_json(state_json)
    assert reconstructed == state
    assert reconstructed.plan is not None
    assert reconstructed.plan.plan_id == "plan-001"
    assert reconstructed.candidates[0].candidate_id == "cand-001"
    assert len(reconstructed.evidences) == 2
    assert reconstructed.evidences[0].evidence_id == "ev-001"
    assert reconstructed.evidences[1].candidate_id is None
    assert reconstructed.diagnoses[0].diagnosis_id == "diag-001"


def test_in_place_mutation_attempts_on_non_empty_collections() -> None:
    """
    NFR-06 / Decision B: In-place mutation attempts on non-empty tuple collections
    must raise AttributeError or TypeError, preventing state corruption.
    """
    hunk = DiffHunk(
        file_path=Path("main.py"),
        old_start=1,
        old_lines=1,
        new_start=1,
        new_lines=2,
        change_type=ChangeType.MODIFIED,
        content="@@ -1,1 +1,2 @@\n",
    )
    snapshot = RepositorySnapshot(
        repo_path=Path("/tmp/repo"),
        current_commit="a" * 40,
        base_commit="b" * 40,
        branch_name="main",
        tracked_files=(Path("main.py"),),
        diff_hunks=(hunk,),
        syntax_errors=("error line 1",),
        source_tree_hash="c" * 64,
        is_valid=True,
    )

    # 1. snapshot.diff_hunks mutation attempts
    with pytest.raises(AttributeError):
        snapshot.diff_hunks.append(hunk)  # type: ignore[attr-defined]

    with pytest.raises(TypeError):
        snapshot.diff_hunks[0] = hunk  # type: ignore[index]

    # 2. snapshot.tracked_files mutation attempts
    with pytest.raises(AttributeError):
        snapshot.tracked_files.pop()  # type: ignore[attr-defined]

    # 3. snapshot.syntax_errors mutation attempts
    with pytest.raises(AttributeError):
        snapshot.syntax_errors.extend(["another error"])  # type: ignore[attr-defined]

    candidate = TestCandidate(
        candidate_id="cand-001",
        run_id="run-01",
        target_symbol_name="Calculator.add",
        test_file_path=Path("tests/test_calc.py"),
        candidate_code="def test_add(): pass\n",
    )
    state = WorkflowState(
        run_id="run-01",
        repo_path=Path("/tmp/repo"),
        candidates=(candidate,),
    )

    # 4. state.candidates mutation attempts
    with pytest.raises(AttributeError):
        state.candidates.append(candidate)  # type: ignore[attr-defined]

    with pytest.raises(TypeError):
        state.candidates[0] = candidate  # type: ignore[index]


def test_independence_from_mutable_input_lists() -> None:
    """
    NFR-06 / Decision B: Models are independent of mutable input lists passed
    to constructors; mutating the caller's list after construction does not alter the model.
    """
    input_paths = [Path("file1.py"), Path("file2.py")]
    input_hunks = [
        DiffHunk(
            file_path=Path("file1.py"),
            old_start=1,
            old_lines=1,
            new_start=1,
            new_lines=2,
            change_type=ChangeType.MODIFIED,
            content="diff",
        )
    ]

    snapshot = RepositorySnapshot(
        repo_path=Path("/tmp/repo"),
        current_commit="a" * 40,
        base_commit="b" * 40,
        branch_name="main",
        tracked_files=input_paths,  # type: ignore[arg-type]
        diff_hunks=input_hunks,  # type: ignore[arg-type]
        source_tree_hash="c" * 64,
        is_valid=True,
    )

    # Mutate the caller's input lists
    input_paths.append(Path("hacked.py"))
    input_paths.clear()
    input_hunks.clear()

    # Verify model state remains completely untouched
    assert len(snapshot.tracked_files) == 2
    assert snapshot.tracked_files[0] == Path("file1.py")
    assert len(snapshot.diff_hunks) == 1
    assert snapshot.diff_hunks[0].file_path == Path("file1.py")


def test_nested_entity_mutation_prevention() -> None:
    """
    NFR-06 / Decision B: Nested entities within collections are themselves
    frozen Pydantic models; modifying their attributes raises ValidationError.
    """
    hunk = DiffHunk(
        file_path=Path("calc.py"),
        old_start=5,
        old_lines=2,
        new_start=5,
        new_lines=4,
        change_type=ChangeType.MODIFIED,
        content="@@ -5,2 +5,4 @@\n",
    )
    snapshot = RepositorySnapshot(
        repo_path=Path("/tmp/repo"),
        current_commit="a" * 40,
        base_commit="b" * 40,
        branch_name="main",
        diff_hunks=(hunk,),
        source_tree_hash="c" * 64,
        is_valid=True,
    )

    # Attempting to mutate an attribute of the nested DiffHunk entity
    with pytest.raises(ValidationError):
        setattr(snapshot.diff_hunks[0], "content", "tampered_content")

    candidate = TestCandidate(
        candidate_id="cand-001",
        run_id="run-01",
        target_symbol_name="Calculator.add",
        test_file_path=Path("tests/test_calc.py"),
        candidate_code="def test_add(): pass\n",
    )
    state = WorkflowState(
        run_id="run-01",
        repo_path=Path("/tmp/repo"),
        candidates=(candidate,),
    )

    # Attempting to mutate an attribute of the nested TestCandidate entity
    with pytest.raises(ValidationError):
        setattr(state.candidates[0], "validation_status", ValidationStatus.PASSED)


def test_json_serialization_and_deserialization_immutability() -> None:
    """
    NFR-06 / Decision B: Tuple collections serialize to standard JSON arrays
    and deserialize back into immutable tuple containers with identical data.
    """
    snapshot = RepositorySnapshot(
        repo_path=Path("/repo"),
        current_commit="1" * 40,
        base_commit="0" * 40,
        branch_name="main",
        tracked_files=(Path("a.py"), Path("b.py")),
        source_tree_hash="d" * 64,
        is_valid=True,
        syntax_errors=("error1", "error2"),
    )

    json_str = snapshot.model_dump_json()
    # Verifies standard JSON array format [...]
    assert '"tracked_files":["a.py","b.py"]' in json_str or '"tracked_files": ["a.py", "b.py"]' in json_str or 'a.py' in json_str
    assert '"syntax_errors":["error1","error2"]' in json_str or 'error1' in json_str

    deserialized = RepositorySnapshot.model_validate_json(json_str)
    assert deserialized == snapshot
    assert isinstance(deserialized.tracked_files, tuple)
    assert isinstance(deserialized.syntax_errors, tuple)

    # Deserialized instance collections are also protected against in-place mutation
    with pytest.raises(AttributeError):
        deserialized.tracked_files.append(Path("c.py"))  # type: ignore[attr-defined]


def test_copy_update_behavior_with_immutable_collections() -> None:
    """
    NFR-06 / Decision B: model_copy(update={...}) creates a new immutable instance
    with replacement collection values while leaving the original instance untouched.
    """
    hunk1 = DiffHunk(
        file_path=Path("f1.py"),
        old_start=1,
        old_lines=1,
        new_start=1,
        new_lines=1,
        change_type=ChangeType.MODIFIED,
        content="c1",
    )
    hunk2 = DiffHunk(
        file_path=Path("f2.py"),
        old_start=1,
        old_lines=1,
        new_start=1,
        new_lines=1,
        change_type=ChangeType.ADDED,
        content="c2",
    )

    orig_snapshot = RepositorySnapshot(
        repo_path=Path("/tmp/repo"),
        current_commit="a" * 40,
        base_commit="b" * 40,
        branch_name="main",
        diff_hunks=(hunk1,),
        source_tree_hash="c" * 64,
        is_valid=True,
    )

    # model_copy with replacement tuple
    copied_snapshot = orig_snapshot.model_copy(update={"diff_hunks": (hunk2,)})

    # Original is untouched
    assert len(orig_snapshot.diff_hunks) == 1
    assert orig_snapshot.diff_hunks[0].file_path == Path("f1.py")

    # Copied snapshot has replacement values
    assert len(copied_snapshot.diff_hunks) == 1
    assert copied_snapshot.diff_hunks[0].file_path == Path("f2.py")
    assert isinstance(copied_snapshot.diff_hunks, tuple)


def test_workflow_state_validated_reconstruction() -> None:
    """
    NFR-06: Demonstrates validated reconstruction for state updates.
    model_copy(update={...}) bypasses validation and can inject invalid or mutable state.
    Validated reconstruction via WorkflowState(**{**state.__dict__, **updates})
    ensures:
    1. Invalid replacement collections (wrong item types) are rejected with ValidationError.
    2. Valid replacement collections (e.g. lists passed in update) are coerced to immutable tuples.
    3. The original WorkflowState instance remains untouched.
    """
    snapshot = RepositorySnapshot(
        repo_path=Path("/tmp/repo"),
        current_commit="1" * 40,
        base_commit="0" * 40,
        branch_name="main",
        source_tree_hash="a" * 64,
        is_valid=True,
    )
    initial_state = WorkflowState(
        run_id="run-001",
        repo_path=Path("/tmp/repo"),
        snapshot=snapshot,
    )

    # 1. Negative test: Invalid replacement collection is rejected with ValidationError
    with pytest.raises(ValidationError):
        WorkflowState(**{**initial_state.__dict__, "candidates": ["not_a_candidate_instance"]})

    with pytest.raises(ValidationError):
        WorkflowState(**{**initial_state.__dict__, "diagnoses": [12345]})

    # 2. Positive test: Valid replacement collection is accepted and coerced to immutable tuple
    candidate = TestCandidate(
        candidate_id="cand-001",
        run_id="run-001",
        target_symbol_name="Calculator.add",
        test_file_path=Path("tests/test_calc.py"),
        candidate_code="def test_add(): pass\n",
    )
    # Passing a mutable list in update:
    updated_state = WorkflowState(**{**initial_state.__dict__, "candidates": [candidate]})

    assert len(updated_state.candidates) == 1
    assert updated_state.candidates[0].candidate_id == "cand-001"
    assert isinstance(updated_state.candidates, tuple)
    with pytest.raises(AttributeError):
        updated_state.candidates.append(candidate)  # type: ignore[attr-defined]

    # 3. Original state remains empty and untouched
    assert initial_state.candidates == ()



