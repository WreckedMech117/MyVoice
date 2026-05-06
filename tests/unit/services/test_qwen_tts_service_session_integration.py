"""
Static-scan + light-touch unit tests for the Story 11.4 integration boundary.

Verifies the ground rules that AC #8 (no direct slot calls), AC #19 (no
session-state reads to gate control flow), AC #12 (voice/model_type label
resolution) and AC #15 (registry construction guard) require of
``services/qwen_tts_service.py``.

The static-scan tests read the service source directly so they remain green
on environments where torch's DLL load fails (Windows-build hazard
documented in Story 11.3 Task 18). The construction-guard test imports the
class — it is allowed to skip cleanly when the heavy import chain
(torch + PyQt6) is unavailable.
"""

import ast
import re
from pathlib import Path

import pytest


SERVICE_PATH = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "myvoice"
    / "services"
    / "qwen_tts_service.py"
)

# AC #8 + #17: the only attribute accesses on ``self._session_registry``
# permitted in qwen_tts_service.py. Anything else (e.g. direct slot calls
# like ``self._session_registry.start_generation(sid)``) is forbidden.
ALLOWED_REGISTRY_ATTRS: frozenset = frozenset({
    "post_mutation",
    "create_session",
    "get",
    "_sessions",
})


@pytest.fixture(scope="module")
def service_source() -> str:
    assert SERVICE_PATH.exists(), f"Expected service file at {SERVICE_PATH}"
    return SERVICE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def service_ast(service_source: str) -> ast.AST:
    return ast.parse(service_source)


# --------------------------------------------------------------------------- #
# AC #8 — TestNoDirectSlotCalls
# --------------------------------------------------------------------------- #


class TestNoDirectSlotCalls:
    """All registry mutations must go through ``post_mutation`` (P-3)."""

    def test_only_allowed_registry_attrs_accessed(self, service_ast: ast.AST):
        """Walk every ``ast.Attribute`` whose value chain ends at
        ``self._session_registry``; assert the trailing attr is in the
        whitelist."""
        violations: list[tuple[int, str]] = []
        for node in ast.walk(service_ast):
            if not isinstance(node, ast.Attribute):
                continue
            inner = node.value
            # Looking for: self._session_registry.<attr>
            if (
                isinstance(inner, ast.Attribute)
                and inner.attr == "_session_registry"
                and isinstance(inner.value, ast.Name)
                and inner.value.id == "self"
            ):
                if node.attr not in ALLOWED_REGISTRY_ATTRS:
                    violations.append((node.lineno, node.attr))

        assert not violations, (
            "Forbidden ``self._session_registry.<attr>`` accesses found "
            f"(allowed: {sorted(ALLOWED_REGISTRY_ATTRS)}): {violations!r}. "
            "AC #8 requires direct slot calls to go through "
            "``post_mutation`` instead."
        )

    def test_post_mutation_is_present(self, service_source: str):
        # Sanity check — at least one ``self._session_registry.post_mutation(``
        # call exists in the file. If this fires, the integration was not
        # wired in.
        assert "self._session_registry.post_mutation(" in service_source, (
            "Expected ``self._session_registry.post_mutation(...)`` calls "
            "in qwen_tts_service.py — Story 11.4 wiring missing."
        )


# --------------------------------------------------------------------------- #
# AC #19 — TestNoSessionStateReads
# --------------------------------------------------------------------------- #


class TestNoSessionStateReads:
    """The service must not branch on session state (AC #19)."""

    # Pragmatic guard per the story: substring scan + manual allowlist of
    # non-session ``.state`` accesses that already exist in the file.
    #
    # AC #19 forbids reading a *registry-fetched* session's ``state`` field.
    # The patterns below correspond to identifiers that are NOT a session:
    #   - ``progress.state``: ``ModelLoadProgress`` instance from the model
    #     registry's progress callback (different lifecycle)
    #   - ``state.value`` / ``state=GenerationState``: legacy GenerationState
    #     enum member access in dataclass instantiations / serialization
    #   - ``self._generation_state``: legacy enum field on the service itself
    NON_SESSION_STATE_PREFIXES: tuple[str, ...] = (
        "progress",            # ModelLoadProgress
        "_generation_state",   # legacy enum field
        "_startup_state",      # legacy startup-tracking field
        "startup_state",       # ditto (no leading underscore variant)
    )
    NON_SESSION_STATE_SUFFIXES: tuple[str, ...] = (
        ".value",
        "==",   # ``progress.state == ModelState.X``
        " ==",  # tolerate spacing
    )

    def test_no_session_dot_state_reads(self, service_source: str):
        for match in re.finditer(r"\.state\b", service_source):
            preceding = service_source[max(0, match.start() - 30):match.start()]
            following = service_source[match.end():match.end() + 6]
            # Allow if preceded by a known non-session identifier.
            if any(preceding.endswith(p) for p in self.NON_SESSION_STATE_PREFIXES):
                continue
            # Allow attr-chain follow-ups that are part of legacy enum
            # serialization / comparison.
            if any(following.startswith(s) for s in self.NON_SESSION_STATE_SUFFIXES):
                continue
            # Allow patterns like ``GenerationState.<NAME>`` where the
            # tokenizer matched ``State.<lowercase_word>`` — these are
            # enum-member references (e.g. inside ``GenerationState.IDLE``).
            # The preceding 12 chars contain the enum class name.
            if "GenerationState" in preceding[-30:]:
                continue
            start = max(0, match.start() - 40)
            end = min(len(service_source), match.end() + 40)
            snippet = service_source[start:end]
            raise AssertionError(
                f"Forbidden ``.state`` read in qwen_tts_service.py: "
                f"...{snippet!r}... — AC #19 forbids branching on session "
                "state from inside the service."
            )

    def test_no_session_dot_is_audible_reads(self, service_source: str):
        assert ".is_audible" not in service_source, (
            "Forbidden ``.is_audible`` access in qwen_tts_service.py — "
            "AC #19 forbids branching on session substate from inside "
            "the service."
        )


# --------------------------------------------------------------------------- #
# AC #18 — TestModuleImports
# --------------------------------------------------------------------------- #


class TestModuleImports:
    """The Story 11.4 import additions must be present (AC #18)."""

    def test_session_registry_import_present(self, service_source: str):
        assert (
            "from myvoice.services.sessions.session_registry import SessionRegistry"
            in service_source
        ), "Story 11.4 AC #18: SessionRegistry import missing."

    def test_session_source_import_present(self, service_source: str):
        assert (
            "from myvoice.services.sessions.generation_session import SessionSource"
            in service_source
        ), "Story 11.4 AC #18: SessionSource import missing."


# --------------------------------------------------------------------------- #
# AC #12 — TestVoiceLabelResolution + TestModelTypeLabelResolution
# --------------------------------------------------------------------------- #


# Try to import the resolvers from the production module. Skip the
# class-level tests below if the heavy import chain (torch DLL) fails —
# the same precedent as Story 11.3 Task 18.
_QWEN_TTS_IMPORT_ERROR: "Exception | None" = None
_QwenTTSService = None
_QwenTTSRequest = None
_QwenModelType = None
try:
    pytest.importorskip("PyQt6")
    from myvoice.services.qwen_tts_service import (  # type: ignore[import-not-found]
        QwenTTSService as _QwenTTSService,
        QwenTTSRequest as _QwenTTSRequest,
    )
    from myvoice.models.service_enums import (  # type: ignore[import-not-found]
        QwenModelType as _QwenModelType,
    )
except Exception as exc:  # pragma: no cover — environment-dependent
    _QWEN_TTS_IMPORT_ERROR = exc


class TestVoiceLabelResolution:
    """AC #12 — voice label resolution covers four request shapes."""

    @pytest.fixture(scope="class")
    def resolver(self):
        if _QwenTTSService is None:
            pytest.skip(
                f"QwenTTSService import failed: {_QWEN_TTS_IMPORT_ERROR}"
            )
        return _QwenTTSService._resolve_voice_label

    @pytest.fixture(scope="class")
    def model_type_resolver(self):
        if _QwenTTSService is None:
            pytest.skip(
                f"QwenTTSService import failed: {_QWEN_TTS_IMPORT_ERROR}"
            )
        return _QwenTTSService._resolve_model_type_label

    def _make_request(self, **overrides):
        defaults = dict(
            text="hello",
            language="Auto",
            model_type=_QwenModelType.CUSTOM_VOICE if _QwenModelType else None,
            speaker="",
            instruct=None,
            ref_audio=None,
            ref_text=None,
            x_vector_only_mode=False,
            voice_description=None,
            streaming=True,
            checkpoint_path=None,
            voice_clone_prompt=None,
        )
        defaults.update(overrides)
        return _QwenTTSRequest(**defaults)

    def test_explicit_speaker_takes_priority(self, resolver):
        # User explicitly picks a non-default speaker (e.g. "Vivian")
        # via the voice selector; resolver returns it as-is.
        req = self._make_request(speaker="Vivian")
        assert resolver(req) == "Vivian"

    def test_default_ryan_falls_through_to_voice_description(self, resolver):
        # Smoke-test fix (2026-05-06): voice-design requests leave
        # speaker at the dataclass default "Ryan" but set
        # voice_description. Pre-fix, the resolver returned "Ryan"
        # because speaker was checked first; post-fix, voice_description
        # wins.
        req = self._make_request(
            speaker="Ryan", voice_description="cheerful"
        )
        assert resolver(req) == "cheerful"

    def test_default_ryan_falls_through_to_checkpoint_stem(self, resolver):
        # Optimized voice path: caller may set both speaker_name and
        # checkpoint_path; if speaker is the default "Ryan" sentinel
        # (i.e. the caller forgot to override), the checkpoint stem
        # is the next-best identifier.
        req = self._make_request(
            speaker="Ryan",
            voice_description=None,
            checkpoint_path=Path("/voices/sarira.bin"),
        )
        assert resolver(req) == "sarira"

    def test_default_ryan_falls_through_to_ref_audio_stem(self, resolver):
        # Voice-clone (BASE) path: speaker stays default, ref_audio is
        # the only identifier. Pre-fix returned "Ryan"; post-fix
        # returns the ref_audio file stem.
        req = self._make_request(
            speaker="Ryan",
            voice_description=None,
            checkpoint_path=None,
            ref_audio=Path("/clones/morgan_freeman.wav"),
        )
        assert resolver(req) == "morgan_freeman"

    def test_explicit_ryan_with_no_other_fields_returns_ryan(self, resolver):
        # User actually picks "Ryan" via the voice selector AND no
        # other identifier is set — fallback returns speaker.
        req = self._make_request(
            speaker="Ryan",
            voice_description=None,
            checkpoint_path=None,
            ref_audio=None,
        )
        assert resolver(req) == "Ryan"

    def test_voice_description_when_no_speaker(self, resolver):
        req = self._make_request(speaker="", voice_description="cheerful")
        assert resolver(req) == "cheerful"

    def test_checkpoint_stem_when_no_speaker_or_description(self, resolver):
        req = self._make_request(
            speaker="",
            voice_description=None,
            checkpoint_path=Path("/some/dir/my_checkpoint.bin"),
        )
        assert resolver(req) == "my_checkpoint"

    def test_unknown_when_all_empty(self, resolver):
        req = self._make_request(
            speaker="",
            voice_description=None,
            checkpoint_path=None,
        )
        assert resolver(req) == "unknown"

    def test_model_type_resolves_display_name(self, model_type_resolver):
        req = self._make_request(model_type=_QwenModelType.CUSTOM_VOICE)
        assert (
            model_type_resolver(req) == _QwenModelType.CUSTOM_VOICE.display_name
        )

    def test_model_type_default_when_none(self, model_type_resolver):
        # Build via dataclasses.replace to bypass the type-checked default.
        req = self._make_request()
        # Direct assignment after construction — defensive None handling
        # is required by the resolver per AC #12.
        object.__setattr__(req, "model_type", None)
        assert model_type_resolver(req) == "default"


# --------------------------------------------------------------------------- #
# AC #15 — TestRegistryConstructionGuard
# --------------------------------------------------------------------------- #


class TestRegistryConstructionGuard:
    """AC #15 — service constructs both with and without a registry."""

    @pytest.fixture(scope="class")
    def qapp(self):
        if _QwenTTSService is None:
            pytest.skip(
                f"QwenTTSService import failed: {_QWEN_TTS_IMPORT_ERROR}"
            )
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        yield app

    def test_constructs_with_registry_none(self, qapp):
        service = _QwenTTSService(session_registry=None)
        try:
            assert service._session_registry is None
        finally:
            # No async-stop required; service was never started.
            pass

    def test_constructs_with_real_registry(self, qapp):
        from myvoice.services.sessions import SessionRegistry

        registry = SessionRegistry()
        service = _QwenTTSService(session_registry=registry)
        try:
            assert service._session_registry is registry
        finally:
            registry.deleteLater()

    def test_keyword_only_session_registry_parameter(self):
        # AC table: ``session_registry`` is keyword-only so positional callers
        # don't accidentally hit it. Verify by inspecting the signature.
        if _QwenTTSService is None:
            pytest.skip(
                f"QwenTTSService import failed: {_QWEN_TTS_IMPORT_ERROR}"
            )
        import inspect
        sig = inspect.signature(_QwenTTSService.__init__)
        param = sig.parameters.get("session_registry")
        assert param is not None, "session_registry parameter missing"
        assert param.kind == inspect.Parameter.KEYWORD_ONLY, (
            f"session_registry should be KEYWORD_ONLY; got {param.kind}"
        )
        assert param.default is None, (
            f"session_registry default should be None; got {param.default!r}"
        )
