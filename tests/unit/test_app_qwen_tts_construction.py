"""Story 18.3 H1 regression test — app.py production wire-up of app_settings
into QwenTTSService.

The bug this catches: ``app.py:_initialize_services`` (or wherever the
``QwenTTSService(...)`` constructor is called from) drops the
``app_settings=self._app_settings`` keyword argument. Without that kwarg,
``QwenTTSService.__init__`` defaults ``app_settings=None``, which routes
``ModelRegistry.__init__`` through the legacy ``dtype`` parameter
mapping and prints ``precision_source='legacy_constructor_arg'`` in
the log — silently disabling Story 18.3's entire user surface
(NFR7 fp32 fallback inert; tts_precision setting ignored; Task 7
NFR1 measurement compares two identical bf16 runs).

The previous wire-up tests at
``test_qwen_tts_service_dispatch.py::test_app_settings_tts_precision_*``
exercise the QwenTTSService → ModelRegistry hop only (because
``_make_service`` always passes ``app_settings=`` itself); they do NOT
catch the production call-site dropping the kwarg. This test is the
surface that catches THAT specific bug class.

Per ``memory/code_review_regression_test_exact_class.md``, the regression
test must mirror the exact bug class — here, "production call site
drops the app_settings keyword argument" — by AST-scanning the actual
construction call in ``app.py`` and asserting the kwarg is present.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


APP_PY_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "myvoice"
    / "app.py"
)


@pytest.fixture(scope="module")
def app_source() -> str:
    assert APP_PY_PATH.exists(), f"Expected app.py at {APP_PY_PATH}"
    return APP_PY_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def app_ast(app_source: str) -> ast.AST:
    return ast.parse(app_source)


def _find_qwen_tts_service_calls(tree: ast.AST) -> list[ast.Call]:
    """Locate every ``QwenTTSService(...)`` constructor call in app.py."""
    calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Match either ``QwenTTSService(...)`` or ``something.QwenTTSService(...)``
        # — the first form is the canonical pattern.
        if isinstance(func, ast.Name) and func.id == "QwenTTSService":
            calls.append(node)
        elif isinstance(func, ast.Attribute) and func.attr == "QwenTTSService":
            calls.append(node)
    return calls


def test_app_py_constructs_qwen_tts_service_at_least_once(app_ast: ast.AST):
    """Sanity check — there must be at least one QwenTTSService constructor
    call in app.py. If this fails, the test's AST scan is searching the
    wrong module (e.g., production code moved to a new file)."""
    calls = _find_qwen_tts_service_calls(app_ast)
    assert calls, (
        "Expected at least one QwenTTSService(...) constructor call in app.py; "
        "found none. Has the construction site moved? Update this test."
    )


def test_app_py_passes_app_settings_to_qwen_tts_service(app_ast: ast.AST):
    """Story 18.3 H1 — every QwenTTSService(...) construction in app.py
    must pass ``app_settings=`` as a keyword argument so the Story 18.3
    precision resolver engages.

    Bug class regression: app.py:_initialize_services dropped this kwarg
    on the initial Story 18.3 source-tree pass; the omission was caught
    only when the dtype-audit log surfaced
    ``precision_source='legacy_constructor_arg'`` instead of
    ``app_settings_auto_ampere``. This test catches the omission at
    static-scan time so the next regression cannot ship silently.
    """
    calls = _find_qwen_tts_service_calls(app_ast)
    assert calls, "no QwenTTSService(...) calls found in app.py"

    violations: list[tuple[int, list[str]]] = []
    for call in calls:
        keyword_names = [kw.arg for kw in call.keywords if kw.arg is not None]
        if "app_settings" not in keyword_names:
            violations.append((call.lineno, keyword_names))

    assert not violations, (
        f"Story 18.3 H1 violated: every QwenTTSService(...) call in app.py "
        f"must pass app_settings= as a kwarg so the precision resolver "
        f"engages. Violations (line, kwargs_seen):\n"
        + "\n".join(f"  line {ln}: {kw}" for ln, kw in violations)
        + "\n\n"
        f"Without app_settings, ModelRegistry routes through the legacy "
        f"dtype path (precision_source='legacy_constructor_arg') and "
        f"the entire Story 18.3 user surface is silently a no-op."
    )


def test_app_py_app_settings_kwarg_value_is_self_app_settings(app_source: str):
    """Stricter check on the exact wire-up surface.

    The kwarg's value must be ``self._app_settings`` specifically (not
    some other variable, not literal ``None``). Pinning the value
    catches a subtler bug class: a future maintainer renames the
    instance attribute but forgets to update the kwarg site.
    """
    # Use a regex-tolerant string check on the source. AST inspection of
    # the value would also work but the string check is more stable
    # against minor formatting changes.
    assert "app_settings=self._app_settings" in app_source, (
        "Expected the literal substring 'app_settings=self._app_settings' "
        "in app.py — the kwarg's value must reference the loaded app "
        "settings instance attribute. If a future refactor renames "
        "_app_settings, both the kwarg site and this test must be "
        "updated together."
    )
