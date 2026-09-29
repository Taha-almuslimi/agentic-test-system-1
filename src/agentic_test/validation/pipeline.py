"""Multi-gate candidate validation pipeline."""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from agentic_test.core.models import TestCandidate, ValidationStatus
from agentic_test.core.protocols.sandbox import SandboxManager
from agentic_test.validation.base import BaseValidator
from agentic_test.validation.collection import CollectionValidator
from agentic_test.validation.models import ValidationResult
from agentic_test.validation.security import SecurityValidator
from agentic_test.validation.syntax import SyntaxValidator


class ValidationPipeline:
    """
    Orchestrates sequential validation gates (Syntax, Security, and optional Collection).
    Applies short-circuit evaluation: fails at the first broken gate.
    Does NOT invoke GenerationService or execute repair loops.

    IMPORTANT SAFETY NOTICE (INV-01):
    Gate 3 (pytest collection) is included only when both sandbox_manager and
    source_root are supplied. It then runs inside an isolated Docker container,
    never in the host process. A default pipeline remains static-only and must
    not be used as authorization for dynamic test execution.
    """

    def __init__(
        self,
        validators: Optional[Sequence[BaseValidator]] = None,
        sandbox_manager: Optional[SandboxManager] = None,
        source_root: Optional[Path] = None,
        collection_staging_root: Optional[Path] = None,
    ) -> None:
        if (sandbox_manager is None) != (source_root is None):
            raise ValueError(
                "sandbox_manager and source_root must be supplied together for Gate 3 collection."
            )

        if validators is not None:
            self._validators: List[BaseValidator] = list(validators)
        else:
            self._validators = [
                SyntaxValidator(),
                SecurityValidator(),
            ]

        if sandbox_manager is not None and source_root is not None:
            self._validators.append(
                CollectionValidator(
                    sandbox_manager=sandbox_manager,
                    source_root=source_root,
                    collection_staging_root=collection_staging_root,
                )
            )

    @property
    def validators(self) -> List[BaseValidator]:
        return self._validators

    @staticmethod
    def _reconstruct_candidate(
        candidate: TestCandidate, status: ValidationStatus
    ) -> TestCandidate:
        """
        Reconstructs a TestCandidate with an updated validation_status via validated
        instantiation to prevent validation bypass (such as from model_copy).
        Preserves all candidate fields, identity, retry_count, and quarantine_reason.
        """
        return TestCandidate(
            **{**candidate.__dict__, "validation_status": status}
        )

    def validate_candidate(
        self, candidate: TestCandidate
    ) -> Tuple[TestCandidate, ValidationResult]:
        """
        Runs candidate through configured validation gates sequentially.
        Short-circuits upon first gate failure.
        """
        # If candidate was already quarantined during generation, do not validate
        if candidate.validation_status == ValidationStatus.QUARANTINED:
            quarantine_result = ValidationResult(
                candidate_id=candidate.candidate_id,
                status=ValidationStatus.QUARANTINED,
                gate="PRE_VALIDATION",
                passed=False,
                error_message=candidate.quarantine_reason or "Candidate quarantined prior to validation",
            )
            return candidate, quarantine_result

        for validator in self._validators:
            result = validator.validate(candidate)
            if not result.passed:
                # Update candidate validation status to match rejected gate via validated reconstruction
                updated_candidate = self._reconstruct_candidate(candidate, result.status)
                return updated_candidate, result

        # All configured gates passed. Preserve the Gate 3 result when it ran so
        # callers can distinguish sandbox collection from static-only screening.
        passed_result = ValidationResult(
            candidate_id=candidate.candidate_id,
            status=ValidationStatus.PASSED,
            gate="STATIC_PIPELINE",
            passed=True,
        )
        updated_candidate = self._reconstruct_candidate(candidate, ValidationStatus.PASSED)
        if self._validators and self._validators[-1].gate_name == "GATE_3_COLLECTION":
            return updated_candidate, result
        return updated_candidate, passed_result

    def validate_candidates(
        self, candidates: Sequence[TestCandidate]
    ) -> List[Tuple[TestCandidate, ValidationResult]]:
        """
        Validates a sequence of test candidates.
        """
        return [self.validate_candidate(candidate) for candidate in candidates]
