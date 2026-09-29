"""
Workflow state container for the Agentic Test Generation System.
Adheres strictly to Stage 3 Section 4.4.3 and Decision 3.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from agentic_test.core.models import (
    ExecutionEvidence,
    ExecutionPlan,
    FailureDiagnosis,
    RepositorySnapshot,
    TestCandidate,
)


class WorkflowState(BaseModel):
    """
    Immutable workflow state capturing point-in-time state transitions across the lifecycle.
    Stage 3 Section 4.4.3 + Decision B immutable container clarification (Tuple[T, ...]).
    """
    model_config = ConfigDict(frozen=True)

    # --- Run Identification & Execution Context ---
    run_id: str
    repo_path: Path
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- Populated during Ingestion & Static Analysis (UC-01) ---
    snapshot: Optional[RepositorySnapshot] = None

    # --- Populated during Deterministic Planning (UC-02) ---
    plan: Optional[ExecutionPlan] = None

    # --- Populated during Test Generation & Multi-Gate Validation (UC-03, UC-04) ---
    candidates: Tuple[TestCandidate, ...] = Field(default_factory=tuple)
    generation_retry_count: int = 0
    max_generation_retries: int = 2

    # --- Populated during Sandbox Execution (UC-05) ---
    evidences: Tuple[ExecutionEvidence, ...] = Field(default_factory=tuple)

    # --- Populated during Failure Diagnosis (UC-06) ---
    diagnoses: Tuple[FailureDiagnosis, ...] = Field(default_factory=tuple)

    # --- Terminal Execution Status ---
    final_status: str = "INITIALIZED"
    error_message: Optional[str] = None
