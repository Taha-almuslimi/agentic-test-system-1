"""
Unit and integration tests for multi-gate static candidate validation.
Verifies Gate 1 Syntax, Gate 2 Security (with strict filesystem write rejection),
and ValidationPipeline short-circuiting.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from agentic_test.core.models import TestCandidate, ValidationStatus
from agentic_test.execution.mock_sandbox import MockSandboxManager
from agentic_test.validation.base import BaseValidator
from agentic_test.validation.models import ValidationResult
from agentic_test.validation.pipeline import ValidationPipeline
from agentic_test.validation.security import SecurityValidator
from agentic_test.validation.syntax import SyntaxValidator


def _make_candidate(
    code: str,
    imports: tuple[str, ...] = ("import pytest",),
    status: ValidationStatus = ValidationStatus.PENDING,
    quarantine_reason: str | None = None,
) -> TestCandidate:
    return TestCandidate(
        candidate_id="tc-val-001",
        run_id="run-val-001",
        target_symbol_name="pkg.calc.add",
        test_file_path=Path("tests/test_calc.py"),
        candidate_code=code,
        imports=imports,
        validation_status=status,
        quarantine_reason=quarantine_reason,
    )


# -------------------------------------------------------------------------
# Gate 1: SyntaxValidator Tests
# -------------------------------------------------------------------------

def test_syntax_validator_valid_code() -> None:
    """Gate 1 passes valid Python syntax."""
    validator = SyntaxValidator()
    cand = _make_candidate("def test_ok():\n    assert 1 + 1 == 2\n")

    result = validator.validate(cand)
    assert result.passed is True
    assert result.status == ValidationStatus.PASSED
    assert result.gate == "GATE_1_SYNTAX"


def test_syntax_validator_invalid_syntax() -> None:
    """Gate 1 rejects malformed Python code with REJECTED_SYNTAX."""
    validator = SyntaxValidator()
    cand = _make_candidate("def test_broken(:\n    assert True")

    result = validator.validate(cand)
    assert result.passed is False
    assert result.status == ValidationStatus.REJECTED_SYNTAX
    assert "SyntaxError" in (result.error_message or "")
    assert len(result.diagnostics) > 0


# -------------------------------------------------------------------------
# Gate 2: SecurityValidator Tests
# -------------------------------------------------------------------------

def test_security_validator_clean_code_passes() -> None:
    """Gate 2 passes safe test code using in-memory mocks."""
    validator = SecurityValidator()
    code = (
        "def test_safe():\n"
        "    from unittest.mock import MagicMock\n"
        "    mock_fn = MagicMock(return_value=42)\n"
        "    assert mock_fn() == 42\n"
    )
    cand = _make_candidate(code)
    result = validator.validate(cand)
    assert result.passed is True
    assert result.status == ValidationStatus.PASSED


def test_security_validator_rejects_forbidden_imports() -> None:
    """Gate 2 rejects forbidden modules (os, subprocess, shutil, socket, etc.)."""
    validator = SecurityValidator()

    forbidden_cases = [
        ("import os", "def test_os(): pass"),
        ("import subprocess", "def test_subp(): pass"),
        ("import shutil", "def test_sh(): pass"),
        ("import socket", "def test_sock(): pass"),
        ("import requests", "def test_req(): pass"),
        ("from os import path", "def test_from_os(): pass"),
        ("from subprocess import run", "def test_from_subp(): pass"),
    ]

    for imp, code in forbidden_cases:
        cand = _make_candidate(code=code, imports=(imp,))
        result = validator.validate(cand)
        assert result.passed is False
        assert result.status == ValidationStatus.REJECTED_SECURITY
        assert "Forbidden" in (result.error_message or "")


def test_security_validator_rejects_forbidden_builtins() -> None:
    """Gate 2 rejects dangerous builtins: eval, exec, compile, __import__."""
    validator = SecurityValidator()

    dangerous_snippets = [
        "def test_eval():\n    eval('2 + 2')\n",
        "def test_exec():\n    exec('x = 1')\n",
        "def test_compile():\n    compile('x = 1', '<string>', 'exec')\n",
        "def test_import():\n    __import__('os')\n",
    ]

    for snippet in dangerous_snippets:
        cand = _make_candidate(code=snippet)
        result = validator.validate(cand)
        assert result.passed is False
        assert result.status == ValidationStatus.REJECTED_SECURITY
        assert "Forbidden builtin call" in (result.error_message or "")


def test_security_validator_allows_safe_methods_and_builtins() -> None:
    """
    Gate 2 allows safe standard library calls and object methods:
    dict.copy(), list.copy(), service.run(), service.call(), getattr(), setattr(),
    and in-memory stream writes (io.StringIO().write()).
    """
    validator = SecurityValidator()

    safe_snippets = [
        "def test_dict_copy():\n    d = {'k': 1}\n    d2 = d.copy()\n    assert d2 == d\n",
        "def test_list_copy():\n    lst = [1, 2, 3]\n    lst2 = lst.copy()\n    assert lst2 == lst\n",
        "def test_service_run():\n    from unittest.mock import MagicMock\n    service = MagicMock()\n    service.run()\n    assert service.run.called\n",
        "def test_service_call():\n    from unittest.mock import MagicMock\n    service = MagicMock()\n    service.call()\n    assert service.call.called\n",
        "def test_getattr():\n    obj = object()\n    assert getattr(obj, '__doc__', None) is not None\n",
        "def test_setattr():\n    class Dummy: pass\n    d = Dummy()\n    setattr(d, 'val', 42)\n    assert d.val == 42\n",
        "def test_stringio_write():\n    import io\n    buf = io.StringIO()\n    buf.write('test data')\n    assert buf.getvalue() == 'test data'\n",
    ]

    for snippet in safe_snippets:
        cand = _make_candidate(code=snippet)
        result = validator.validate(cand)
        assert result.passed is True, f"Failed for safe snippet:\n{snippet}\nError: {result.error_message}"
        assert result.status == ValidationStatus.PASSED


def test_security_validator_rejects_subprocess_calls() -> None:
    """Gate 2 rejects subprocess calls: run, call, Popen, check_call, check_output."""
    validator = SecurityValidator()

    subprocess_snippets = [
        "def test_subp_run():\n    import subprocess\n    subprocess.run(['echo', '1'])\n",
        "def test_subp_call():\n    import subprocess\n    subprocess.call(['echo', '1'])\n",
        "def test_subp_popen():\n    import subprocess\n    subprocess.Popen(['echo', '1'])\n",
        "def test_subp_check_call():\n    import subprocess\n    subprocess.check_call(['echo', '1'])\n",
        "def test_subp_check_output():\n    import subprocess\n    subprocess.check_output(['echo', '1'])\n",
    ]

    for snippet in subprocess_snippets:
        cand = _make_candidate(code=snippet)
        result = validator.validate(cand)
        assert result.passed is False
        assert result.status == ValidationStatus.REJECTED_SECURITY


def test_security_validator_rejects_filesystem_writes_path_write() -> None:
    """Gate 2 rejects Path.write_text and Path.write_bytes."""
    validator = SecurityValidator()

    cand_text = _make_candidate(
        code="def test_write():\n    from pathlib import Path\n    Path('out.txt').write_text('content')\n"
    )
    res_text = validator.validate(cand_text)
    assert res_text.passed is False
    assert res_text.status == ValidationStatus.REJECTED_SECURITY
    assert "Forbidden filesystem mutation 'write_text()'" in (res_text.error_message or "")

    cand_bytes = _make_candidate(
        code="def test_write_b():\n    from pathlib import Path\n    Path('out.bin').write_bytes(b'abc')\n"
    )
    res_bytes = validator.validate(cand_bytes)
    assert res_bytes.passed is False
    assert res_bytes.status == ValidationStatus.REJECTED_SECURITY
    assert "Forbidden filesystem mutation 'write_bytes()'" in (res_bytes.error_message or "")


def test_security_validator_rejects_open_write_modes() -> None:
    """Gate 2 rejects open() with write, append, or read-write modes."""
    validator = SecurityValidator()

    write_snippets = [
        "def test_open_w():\n    with open('file.txt', 'w') as f:\n        pass\n",
        "def test_open_a():\n    with open('file.txt', 'a') as f:\n        pass\n",
        "def test_open_kw():\n    with open('file.txt', mode='w+') as f:\n        pass\n",
        "def test_open_x():\n    with open('file.txt', 'x') as f:\n        pass\n",
    ]

    for snippet in write_snippets:
        cand = _make_candidate(snippet)
        res = validator.validate(cand)
        assert res.passed is False
        assert res.status == ValidationStatus.REJECTED_SECURITY


def test_security_validator_rejects_filesystem_mutations() -> None:
    """Gate 2 rejects explicit filesystem mutation calls: shutil.rmtree, shutil.copyfile, os.remove, os.unlink."""
    validator = SecurityValidator()

    mutation_snippets = [
        "def test_rmtree():\n    import shutil\n    shutil.rmtree('target_dir')\n",
        "def test_copyfile():\n    import shutil\n    shutil.copyfile('a', 'b')\n",
        "def test_remove():\n    import os\n    os.remove('file.txt')\n",
        "def test_unlink():\n    import os\n    os.unlink('file.txt')\n",
    ]

    for snippet in mutation_snippets:
        cand = _make_candidate(snippet)
        res = validator.validate(cand)
        assert res.passed is False
        assert res.status == ValidationStatus.REJECTED_SECURITY


def test_security_validator_rejects_tmp_path_writes_strictly() -> None:
    """
    Gate 2 rejects filesystem writes even when targeting pytest fixtures like tmp_path.
    No exceptions are granted for tmp_path, tempfile, or fixture-derived variables.
    """
    validator = SecurityValidator()

    fixture_snippets = [
        "def test_tmp_path_write_text(tmp_path):\n    p = tmp_path / 'sample.txt'\n    p.write_text('content')\n",
        "def test_tmp_path_open_write(tmp_path):\n    with open(tmp_path / 'sample.txt', 'w') as f:\n        pass\n",
        "def test_tmp_path_mkdir(tmp_path):\n    (tmp_path / 'subdir').mkdir()\n",
    ]

    for snippet in fixture_snippets:
        cand = _make_candidate(snippet)
        res = validator.validate(cand)
        assert res.passed is False, f"Expected rejection for: {snippet}"
        assert res.status == ValidationStatus.REJECTED_SECURITY


# -------------------------------------------------------------------------
# ValidationPipeline Integration Tests
# -------------------------------------------------------------------------

def test_validation_pipeline_short_circuits_on_gate_1_syntax() -> None:
    """Pipeline short-circuits at Gate 1 on syntax error without running Gate 2."""
    pipeline = ValidationPipeline()
    # Code with both syntax error AND security violation (import os)
    cand = _make_candidate(
        code="def test_broken(:\n    import os\n",
        imports=("import os",),
    )

    updated_cand, result = pipeline.validate_candidate(cand)
    assert updated_cand.validation_status == ValidationStatus.REJECTED_SYNTAX
    assert result.gate == "GATE_1_SYNTAX"
    assert result.status == ValidationStatus.REJECTED_SYNTAX


def test_validation_pipeline_fails_at_gate_2_security() -> None:
    """Pipeline passes Gate 1 then fails at Gate 2 for security violation."""
    pipeline = ValidationPipeline()
    # Valid syntax, but imports subprocess
    cand = _make_candidate(
        code="def test_sub():\n    pass\n",
        imports=("import subprocess",),
    )

    updated_cand, result = pipeline.validate_candidate(cand)
    assert updated_cand.validation_status == ValidationStatus.REJECTED_SECURITY
    assert result.gate == "GATE_2_SECURITY"
    assert result.status == ValidationStatus.REJECTED_SECURITY


def test_validation_pipeline_all_gates_pass() -> None:
    """Pipeline passes safe candidate through both gates to ValidationStatus.PASSED."""
    pipeline = ValidationPipeline()
    cand = _make_candidate(
        code="def test_clean():\n    assert 2 + 2 == 4\n",
        imports=("import pytest",),
    )

    updated_cand, result = pipeline.validate_candidate(cand)
    assert updated_cand.validation_status == ValidationStatus.PASSED
    assert result.passed is True
    assert result.status == ValidationStatus.PASSED


def test_validation_pipeline_preserves_pre_quarantined_candidate() -> None:
    """Pre-quarantined candidate from GenerationService is not re-validated."""
    pipeline = ValidationPipeline()
    cand = _make_candidate(
        code="broken",
        status=ValidationStatus.QUARANTINED,
        quarantine_reason="EXHAUSTED_GENERATION_REPAIRS",
    )

    updated_cand, result = pipeline.validate_candidate(cand)
    assert updated_cand.validation_status == ValidationStatus.QUARANTINED
    assert result.passed is False
    assert result.status == ValidationStatus.QUARANTINED
    assert result.gate == "PRE_VALIDATION"


def test_validation_pipeline_gate_3_blocked_invariant() -> None:
    """
    The default pipeline contains ONLY static gates. Gate 3 needs an injected
    isolated sandbox and repository root, so it can never run on the host.
    """
    pipeline = ValidationPipeline()
    gate_names = [v.gate_name for v in pipeline.validators]
    assert gate_names == ["GATE_1_SYNTAX", "GATE_2_SECURITY"]
    assert "GATE_3_COLLECTION" not in gate_names


def test_validation_pipeline_runs_gate_3_collection_in_sandbox(tmp_path: Path) -> None:
    """Gate 3 stages and collects a safe candidate in the isolated sandbox."""
    source_root = tmp_path / "repository"
    source_root.mkdir()
    collection_staging_root = tmp_path / "collection-staging"
    collection_staging_root.mkdir()
    sandbox = MockSandboxManager()
    sandbox.script_success(stdout="collected 1 item\n")
    pipeline = ValidationPipeline(
        sandbox_manager=sandbox,
        source_root=source_root,
        collection_staging_root=collection_staging_root,
    )

    updated_cand, result = pipeline.validate_candidate(
        _make_candidate("def test_clean():\n    assert 2 + 2 == 4\n")
    )

    assert updated_cand.validation_status == ValidationStatus.PASSED
    assert result.passed is True
    assert result.gate == "GATE_3_COLLECTION"
    assert sandbox.call_count == 1
    assert sandbox.calls[0]["command"] == [
        "pytest",
        "--collect-only",
        "/workspace/tests",
        "-o",
        "cache_dir=/tmp/.pytest_cache",
    ]
    assert sandbox.calls[0]["workdir"] == "/workspace"
    config = next(iter(sandbox.created_configs.values()))
    assert config.read_only_mounts[source_root] == "/workspace/src"
    assert "/workspace/tests" in config.read_only_mounts.values()
    assert sandbox.active_containers == []


def test_validation_pipeline_rejects_gate_3_collection_error(tmp_path: Path) -> None:
    """A non-zero pytest collection exit rejects the candidate at Gate 3."""
    source_root = tmp_path / "repository"
    source_root.mkdir()
    collection_staging_root = tmp_path / "collection-staging"
    collection_staging_root.mkdir()
    sandbox = MockSandboxManager()
    sandbox.script_failure(
        exit_code=2,
        stdout="ERROR collecting test_candidate.py\n",
    )
    pipeline = ValidationPipeline(
        sandbox_manager=sandbox,
        source_root=source_root,
        collection_staging_root=collection_staging_root,
    )

    updated_cand, result = pipeline.validate_candidate(
        _make_candidate("def test_clean():\n    assert 2 + 2 == 4\n")
    )

    assert updated_cand.validation_status == ValidationStatus.REJECTED_COLLECTION
    assert result.passed is False
    assert result.gate == "GATE_3_COLLECTION"
    assert "exit code 2" in (result.error_message or "")
    assert result.diagnostics == ("ERROR collecting test_candidate.py",)
    assert sandbox.active_containers == []


def test_validation_pipeline_short_circuits_before_gate_3(tmp_path: Path) -> None:
    """Syntax and security rejection prevents any sandbox collection call."""
    source_root = tmp_path / "repository"
    source_root.mkdir()
    sandbox = MockSandboxManager()
    pipeline = ValidationPipeline(sandbox_manager=sandbox, source_root=source_root)

    updated_cand, result = pipeline.validate_candidate(
        _make_candidate(
            code="def test_unsafe():\n    subprocess.run(['echo', 'unsafe'])\n",
            imports=("import subprocess",),
        )
    )

    assert updated_cand.validation_status == ValidationStatus.REJECTED_SECURITY
    assert result.gate == "GATE_2_SECURITY"
    assert sandbox.call_count == 0


# -------------------------------------------------------------------------
# Validated Reconstruction & Status Transition Invariant Tests
# -------------------------------------------------------------------------

def test_validation_pipeline_path_quarantined_preserved() -> None:
    """
    Path 1: Generation QUARANTINED candidate remains QUARANTINED with reason unchanged.
    Identity, retry count, timestamps, and fields are preserved.
    """
    pipeline = ValidationPipeline()
    cand = _make_candidate(
        code="def unrepairable_code(:",
        status=ValidationStatus.QUARANTINED,
        quarantine_reason="EXHAUSTED_GENERATION_REPAIRS",
    )
    cand_with_retries = cand.model_copy(update={"retry_count": 2})

    updated_cand, result = pipeline.validate_candidate(cand_with_retries)

    assert updated_cand.validation_status == ValidationStatus.QUARANTINED
    assert updated_cand.quarantine_reason == "EXHAUSTED_GENERATION_REPAIRS"
    assert updated_cand.candidate_id == cand_with_retries.candidate_id
    assert updated_cand.run_id == cand_with_retries.run_id
    assert updated_cand.retry_count == 2
    assert updated_cand.created_at == cand_with_retries.created_at
    assert result.status == ValidationStatus.QUARANTINED
    assert result.passed is False
    assert result.gate == "PRE_VALIDATION"


def test_validation_pipeline_path_gate_1_rejected_syntax() -> None:
    """
    Path 2: Gate 1 failure transitions to REJECTED_SYNTAX via validated reconstruction.
    Preserves all fields, identity, retry count, created_at, with quarantine_reason remaining None.
    """
    pipeline = ValidationPipeline()
    cand = _make_candidate(
        code="def test_broken(:\n    assert True\n",
        status=ValidationStatus.PENDING,
    )
    cand_with_retries = cand.model_copy(update={"retry_count": 1})

    updated_cand, result = pipeline.validate_candidate(cand_with_retries)

    assert updated_cand.validation_status == ValidationStatus.REJECTED_SYNTAX
    assert updated_cand.quarantine_reason is None
    assert updated_cand.candidate_id == cand_with_retries.candidate_id
    assert updated_cand.run_id == cand_with_retries.run_id
    assert updated_cand.target_symbol_name == cand_with_retries.target_symbol_name
    assert updated_cand.test_file_path == cand_with_retries.test_file_path
    assert updated_cand.candidate_code == cand_with_retries.candidate_code
    assert updated_cand.imports == cand_with_retries.imports
    assert updated_cand.retry_count == 1
    assert updated_cand.created_at == cand_with_retries.created_at
    assert result.status == ValidationStatus.REJECTED_SYNTAX
    assert result.gate == "GATE_1_SYNTAX"
    assert result.passed is False


def test_validation_pipeline_path_gate_2_rejected_security() -> None:
    """
    Path 3: Gate 2 failure transitions to REJECTED_SECURITY via validated reconstruction.
    Preserves all fields, identity, retry count, created_at, with quarantine_reason remaining None.
    """
    pipeline = ValidationPipeline()
    cand = _make_candidate(
        code="def test_unsafe():\n    import subprocess\n    subprocess.run(['ls'])\n",
        imports=("import subprocess",),
        status=ValidationStatus.PENDING,
    )

    updated_cand, result = pipeline.validate_candidate(cand)

    assert updated_cand.validation_status == ValidationStatus.REJECTED_SECURITY
    assert updated_cand.quarantine_reason is None
    assert updated_cand.candidate_id == cand.candidate_id
    assert updated_cand.run_id == cand.run_id
    assert updated_cand.retry_count == cand.retry_count
    assert updated_cand.created_at == cand.created_at
    assert result.status == ValidationStatus.REJECTED_SECURITY
    assert result.gate == "GATE_2_SECURITY"
    assert result.passed is False


def test_validation_pipeline_path_static_passed() -> None:
    """
    Path 4: Clean candidate transitions to PASSED for static screening only via validated reconstruction.
    Preserves all fields, identity, retry count, created_at, with quarantine_reason remaining None.
    """
    pipeline = ValidationPipeline()
    cand = _make_candidate(
        code="def test_add():\n    assert 1 + 1 == 2\n",
        imports=("import pytest",),
        status=ValidationStatus.PENDING,
    )
    cand_with_retries = cand.model_copy(update={"retry_count": 1})

    updated_cand, result = pipeline.validate_candidate(cand_with_retries)

    assert updated_cand.validation_status == ValidationStatus.PASSED
    assert updated_cand.quarantine_reason is None
    assert updated_cand.candidate_id == cand_with_retries.candidate_id
    assert updated_cand.run_id == cand_with_retries.run_id
    assert updated_cand.retry_count == 1
    assert updated_cand.created_at == cand_with_retries.created_at
    assert result.status == ValidationStatus.PASSED
    assert result.gate == "STATIC_PIPELINE"
    assert result.passed is True


def test_validation_pipeline_reconstruct_candidate_rejects_invalid_enum_status() -> None:
    """
    Negative test: Validated reconstruction strictly rejects invalid enum statuses.
    Ensures that validation bypass (which occurred with model_copy) is impossible.
    """
    cand = _make_candidate(code="def test_foo(): pass\n")

    # 1. Negative test: validated reconstruction raises ValidationError on invalid status
    with pytest.raises(ValidationError):
        ValidationPipeline._reconstruct_candidate(cand, "INVALID_STATUS")  # type: ignore[arg-type]

    with pytest.raises(ValidationError):
        ValidationPipeline._reconstruct_candidate(cand, 999)  # type: ignore[arg-type]

    # 2. Positive test: validated reconstruction succeeds on all valid ValidationStatus enum values
    for valid_status in ValidationStatus:
        reconstructed = ValidationPipeline._reconstruct_candidate(cand, valid_status)
        assert reconstructed.validation_status == valid_status
        assert reconstructed.candidate_id == cand.candidate_id
        assert reconstructed.created_at == cand.created_at
