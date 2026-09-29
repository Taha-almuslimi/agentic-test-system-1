"""
Core domain models and value objects for the Agentic Test Generation System.
Adheres strictly to Stage 2 Section 4.3.3 and the Human-Approved Option A decision.
"""

from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


class RepositoryValidationError(Exception):
    """Raised when repository validation fails (e.g. non-existent or uninitialized directory)."""
    pass


class ChangeType(str, Enum):
    """Classification of code modification types from Git diff hunks."""
    ADDED = "ADDED"
    MODIFIED = "MODIFIED"
    DELETED = "DELETED"


class SymbolType(str, Enum):
    """Structural classification of callable code entities extracted via AST."""
    FUNCTION = "FUNCTION"
    METHOD = "METHOD"
    CLASS = "CLASS"
    MODULE = "MODULE"


class DiffHunk(BaseModel):
    """
    Immutable value object representing a unified diff block.
    Stage 2 Section 4.3.3.
    """
    model_config = ConfigDict(frozen=True)

    file_path: Path
    old_start: int
    old_lines: int
    new_start: int
    new_lines: int
    change_type: ChangeType
    content: str


class SymbolContract(BaseModel):
    """
    Structural specification and interface contract of a callable Python symbol.
    Stage 2 Section 4.3.3.
    """
    model_config = ConfigDict(frozen=True)

    qualified_name: str
    symbol_type: SymbolType
    file_path: Path
    line_range: Tuple[int, int]
    signature: str
    docstring: Optional[str] = None
    dependencies: Tuple[str, ...] = Field(default_factory=tuple)
    is_affected: bool = False


class RepositorySnapshot(BaseModel):
    """
    Immutable point-in-time capture of the target Git repository.
    Stage 2 Section 4.3.3 + Human-Approved Option A Clarification (2026-09-23)
    + Decision B immutable container clarification (Tuple[T, ...]).
    """
    model_config = ConfigDict(frozen=True)

    # Nine baseline fields from Stage 2 Section 4.3.3
    repo_path: Path
    current_commit: str
    base_commit: str
    branch_name: str
    tracked_files: Tuple[Path, ...] = Field(default_factory=tuple)
    existing_test_files: Tuple[Path, ...] = Field(default_factory=tuple)
    source_tree_hash: str
    is_valid: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Approved Option A fields with independent immutable tuple defaults
    diff_hunks: Tuple[DiffHunk, ...] = Field(default_factory=tuple)
    affected_symbols: Tuple[SymbolContract, ...] = Field(default_factory=tuple)
    deleted_symbols: Tuple[SymbolContract, ...] = Field(default_factory=tuple)
    syntax_errors: Tuple[str, ...] = Field(default_factory=tuple)


class WorkflowRoute(str, Enum):
    """Workflow execution routing decisions. Stage 2 §4.3.3."""
    ROUTE_NO_OP = "ROUTE_NO_OP"
    ROUTE_TO_DOCKER_EXECUTION = "ROUTE_TO_DOCKER_EXECUTION"
    ROUTE_TO_TEST_GENERATION = "ROUTE_TO_TEST_GENERATION"


class ValidationStatus(str, Enum):
    """Test candidate validation status. Stage 2 §4.3.3."""
    PENDING = "PENDING"
    PASSED = "PASSED"
    REJECTED_SYNTAX = "REJECTED_SYNTAX"
    REJECTED_SECURITY = "REJECTED_SECURITY"
    REJECTED_COLLECTION = "REJECTED_COLLECTION"
    QUARANTINED = "QUARANTINED"


class FailureCategory(str, Enum):
    """Seven-category failure taxonomy. Stage 2 §4.3.3."""
    TEST_OUTDATED = "TEST_OUTDATED"
    APPLICATION_BUG = "APPLICATION_BUG"
    INVALID_GENERATED_TEST = "INVALID_GENERATED_TEST"
    ENVIRONMENT_FAILURE = "ENVIRONMENT_FAILURE"
    CONFIGURATION_ERROR = "CONFIGURATION_ERROR"
    LLM_FAILURE = "LLM_FAILURE"
    UNKNOWN = "UNKNOWN"


class TriageEngine(str, Enum):
    """Classification engine tier. Stage 2 §4.3.3."""
    DETERMINISTIC_RULE = "DETERMINISTIC_RULE"
    COGNITIVE_LLM = "COGNITIVE_LLM"


class ExecutionPlan(BaseModel):
    """
    Deterministic output of the planning phase.
    Stage 2 Section 4.3.3.
    """
    model_config = ConfigDict(frozen=True)

    plan_id: str
    route: WorkflowRoute
    target_symbols: Tuple[SymbolContract, ...] = Field(default_factory=tuple)
    existing_tests_to_run: Tuple[Path, ...] = Field(default_factory=tuple)
    rationale: str
    decision_hash: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TestCandidate(BaseModel):
    """
    Synthesized unit test candidate and validation state.
    Stage 2 Section 4.3.3.
    """
    model_config = ConfigDict(frozen=True)
    __test__ = False

    candidate_id: str
    run_id: str
    target_symbol_name: str
    test_file_path: Path
    candidate_code: str
    imports: Tuple[str, ...] = Field(default_factory=tuple)
    validation_status: ValidationStatus = ValidationStatus.PENDING
    quarantine_reason: Optional[str] = None
    retry_count: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExecutionEvidence(BaseModel):
    """
    Container execution telemetry and coverage metrics.
    Stage 2 Section 4.3.3.
    """
    model_config = ConfigDict(frozen=True)

    evidence_id: str
    run_id: str
    candidate_id: Optional[str]
    exit_code: int
    stdout: str
    stderr: str
    duration_sec: float
    timed_out: bool = False
    line_coverage: float = 0.0
    branch_coverage: float = 0.0
    line_coverage_delta: Optional[float] = None
    branch_coverage_delta: Optional[float] = None
    traceback: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FailureDiagnosis(BaseModel):
    """
    Diagnostic classification of a failed test candidate.
    Stage 2 Section 4.3.3.
    """
    model_config = ConfigDict(frozen=True)

    diagnosis_id: str
    evidence_id: str
    canonical_category: FailureCategory
    confidence: float
    triage_engine: TriageEngine
    explanation: str
    is_application_bug: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

