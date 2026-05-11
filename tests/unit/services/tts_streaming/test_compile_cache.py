"""Tests for the persistent torch.compile cache helpers (Story 18.4).

Mirrors ``test_torch_runtime.py``'s discipline:

  * Monkeypatches ``os.environ["LOCALAPPDATA"]`` / ``sys.platform`` at the
    attribute-access level so the lazy-stdlib-import discipline inside
    ``compile_cache.py`` is honored without import-order gymnastics.
  * Uses ``tmp_path`` as the cache root for filesystem-touching tests so
    test isolation is preserved — no test leaves state under the real
    ``%LOCALAPPDATA%/MyVoice/torch_compile_cache``.
  * The seven-dimension key uniqueness test uses parametrization to vary
    EACH dimension independently (architecture P-10 enforcement; the bug
    class is "cache-key dimension drift" — a test that only exercises 3 of
    7 dimensions and assumes the rest are "obvious" would be the wrong fix
    class per ``memory/code_review_regression_test_exact_class.md``).

Critical fixture: ``_snapshot_and_restore_torchinductor_env`` captures
``os.environ["TORCHINDUCTOR_CACHE_DIR"]`` before each test and restores it
after. ``set_torchinductor_cache_dir`` mutates that env var as a deliberate
side effect; without snapshot/restore, test ordering would silently shape
the env-var assertions across rows. Mirrors the flag-snapshot/restore
discipline from ``test_torch_runtime.py``.
"""

from __future__ import annotations

import json
import logging
import os
import sys

import pytest

from myvoice.services.tts_streaming import compile_cache


# -- Fixtures ---------------------------------------------------------------- #


@pytest.fixture
def _snapshot_and_restore_torchinductor_env():
    """Snapshot + restore os.environ['TORCHINDUCTOR_CACHE_DIR'].

    ``set_torchinductor_cache_dir`` mutates this env var as a deliberate
    side effect. Tests must be insulated from cross-test bleed.
    """
    snapshot = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    yield snapshot
    if snapshot is None:
        os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
    else:
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = snapshot


@pytest.fixture
def _reset_set_torchinductor_logged_once():
    """Reset the set-once guard so each test sees a fresh first-call.

    The ``_TORCHINDUCTOR_CACHE_DIR_LOGGED_ONCE`` module-level boolean
    survives across tests by default; reset it so the "logs INFO once"
    assertion in the idempotent-set test sees a deterministic first-call
    state.
    """
    snapshot = compile_cache._TORCHINDUCTOR_CACHE_DIR_LOGGED_ONCE
    compile_cache._TORCHINDUCTOR_CACHE_DIR_LOGGED_ONCE = False
    yield
    compile_cache._TORCHINDUCTOR_CACHE_DIR_LOGGED_ONCE = snapshot


@pytest.fixture
def _isolated_cache_root(monkeypatch, tmp_path):
    """Force ``cache_root()`` to return ``tmp_path`` for the duration of a test.

    Monkeypatches ``sys.platform`` to a non-Windows value and ``Path.home``
    to return ``tmp_path`` so the non-Windows branch of ``cache_root()``
    builds the root under the temp directory. Filesystem-touching tests
    use this to keep test state contained.
    """
    monkeypatch.setattr(compile_cache, "sys", _make_platform_module("linux"))
    monkeypatch.setattr(
        "pathlib.Path.home",
        classmethod(lambda cls: tmp_path),
    )
    yield tmp_path


def _make_platform_module(platform: str):
    """Build a thin object that exposes only ``sys.platform`` for the cache module."""

    class _SysShim:
        pass

    shim = _SysShim()
    shim.platform = platform
    # Copy any other sys attributes the module may need (none today, but
    # defensive against future additions).
    return shim


# -- AC #3 row 1: compute_key stability -------------------------------------- #


_BASE_KEY_KWARGS = dict(
    qwen_tts_pin_hash="3fdb4682",
    model_id="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    precision_str="bf16",
    torch_version="2.10.0+cu128",
    decode_window_frames=30,
    cuda_capability=(12, 0),
    compile_mode="reduce-overhead",
)


def test_compute_key_is_stable_across_invocations():
    """P-10 key stability: identical inputs produce identical SHA-256 digest."""
    a = compile_cache.compute_key(**_BASE_KEY_KWARGS)
    b = compile_cache.compute_key(**_BASE_KEY_KWARGS)
    assert a == b
    # Sanity: must be a 64-char hex digest (SHA-256 hexdigest)
    assert len(a) == 64
    assert all(c in "0123456789abcdef" for c in a)


def test_compute_key_is_stable_across_python_processes(tmp_path):
    """P-10 key stability: digest is deterministic across process boundaries.

    Spawn a subprocess that imports ``compile_cache`` directly (NOT via the
    ``myvoice.*`` package, which would trigger the UI + qwen-tts + torch
    import chain — and the subprocess does not get conftest's torch-DLL
    preamble). Defends against accidental nondeterminism (e.g., a future
    refactor that folds ``hash(...)`` randomization into the key, which IS
    randomized via ``PYTHONHASHSEED``).
    """
    import subprocess
    from pathlib import Path as _PathForLocate

    # Direct-import the compile_cache module via importlib without going
    # through the ``myvoice`` package namespace. This sidesteps the
    # ``myvoice/__init__.py`` chain (which imports the entire UI surface
    # and triggers the torch-DLL ordering requirement that conftest.py
    # handles for the test process but not for a fresh subprocess).
    compile_cache_path = _PathForLocate(compile_cache.__file__).resolve()

    script = tmp_path / "compute_key_subprocess.py"
    script.write_text(
        "import importlib.util\n"
        f"_spec = importlib.util.spec_from_file_location('compile_cache', r{str(compile_cache_path)!r})\n"
        "_mod = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_mod)\n"
        "print(_mod.compute_key(\n"
        f"    qwen_tts_pin_hash={_BASE_KEY_KWARGS['qwen_tts_pin_hash']!r},\n"
        f"    model_id={_BASE_KEY_KWARGS['model_id']!r},\n"
        f"    precision_str={_BASE_KEY_KWARGS['precision_str']!r},\n"
        f"    torch_version={_BASE_KEY_KWARGS['torch_version']!r},\n"
        f"    decode_window_frames={_BASE_KEY_KWARGS['decode_window_frames']!r},\n"
        f"    cuda_capability={_BASE_KEY_KWARGS['cuda_capability']!r},\n"
        f"    compile_mode={_BASE_KEY_KWARGS['compile_mode']!r},\n"
        "))\n",
        encoding="utf-8",
    )

    expected = compile_cache.compute_key(**_BASE_KEY_KWARGS)

    # PYTHONHASHSEED=0 defends against any future accidental dependency
    # on Python's randomized hash seed. The current implementation does
    # NOT depend on it (hashlib.sha256 is deterministic), but the test
    # is a contract — a future refactor that violates the contract should
    # fail here.
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(tmp_path.parent),
        timeout=30,
    )
    assert result.returncode == 0, (
        f"subprocess failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    subprocess_key = result.stdout.strip()
    assert subprocess_key == expected


# -- AC #3 row 2: compute_key uniqueness (one parametrized row per dimension) #


@pytest.mark.parametrize(
    "dimension,perturbed_value",
    [
        ("qwen_tts_pin_hash", "1ab0dd75"),  # previous pin
        ("model_id", "Qwen/Qwen3-TTS-12Hz-1.7B-Custom"),  # different checkpoint
        ("precision_str", "fp32"),  # different precision
        ("torch_version", "2.9.0+cu121"),  # downgraded torch
        ("decode_window_frames", 80),  # fork default vs MyVoice's 30
        ("cuda_capability", (8, 9)),  # RTX 40xx vs RTX 5090
        ("compile_mode", "max-autotune"),  # alternate compile mode
    ],
)
def test_compute_key_varies_with_each_of_seven_dimensions(
    dimension, perturbed_value
):
    """P-10 key uniqueness: varying any one of seven dimensions changes the digest.

    The seven parametrized rows enforce the architecture D-24 + P-10
    contract: each dimension must materially affect the key, so a torch
    upgrade / precision toggle / hardware change / pin bump / window
    re-tune / compile-mode change invalidates the cache cleanly.
    """
    baseline = compile_cache.compute_key(**_BASE_KEY_KWARGS)
    perturbed_kwargs = {**_BASE_KEY_KWARGS, dimension: perturbed_value}
    perturbed = compile_cache.compute_key(**perturbed_kwargs)
    assert baseline != perturbed, (
        f"Cache-key dimension {dimension!r} did not affect the digest — "
        f"P-10 violation; cache invalidation would silently fail to fire."
    )


# -- AC #3 row 3: cache_root() Windows path ---------------------------------- #


def test_cache_root_windows_uses_localappdata(monkeypatch, tmp_path):
    """Windows path: ``cache_root()`` returns ``$LOCALAPPDATA/MyVoice/torch_compile_cache``."""
    # Force the Windows branch.
    monkeypatch.setattr(compile_cache, "sys", _make_platform_module("win32"))
    fake_localappdata = tmp_path / "AppData" / "Local"
    monkeypatch.setenv("LOCALAPPDATA", str(fake_localappdata))

    root = compile_cache.cache_root()
    assert root == fake_localappdata / "MyVoice" / "torch_compile_cache"
    assert root.exists()  # mkdir was called


# -- AC #3 row 4: cache_root() non-Windows fallback -------------------------- #


def test_cache_root_non_windows_uses_xdg_style_home(monkeypatch, tmp_path):
    """Non-Windows path: ``cache_root()`` returns ``~/.cache/MyVoice/torch_compile_cache``."""
    monkeypatch.setattr(compile_cache, "sys", _make_platform_module("linux"))
    monkeypatch.setattr(
        "pathlib.Path.home",
        classmethod(lambda cls: tmp_path),
    )

    root = compile_cache.cache_root()
    assert root == tmp_path / ".cache" / "MyVoice" / "torch_compile_cache"
    assert root.exists()


# -- AC #3 row 5: is_warm() / mark_warm() round-trip ------------------------- #


def test_is_warm_round_trip_with_mark_warm(_isolated_cache_root):
    """``mark_warm`` writes meta.json; ``is_warm`` reads it; round-trip is True."""
    key = compile_cache.compute_key(**_BASE_KEY_KWARGS)

    # Cold cache initially.
    assert compile_cache.is_warm(key) is False

    # Mark warm — should write meta.json.
    compile_cache.mark_warm(key)
    assert compile_cache.is_warm(key) is True

    # Sanity: meta.json exists and round-trips through JSON correctly.
    meta_path = compile_cache.cache_dir_for_key(key) / "meta.json"
    assert meta_path.exists()
    with meta_path.open("r", encoding="utf-8") as fp:
        meta = json.load(fp)
    assert meta["key"] == key
    assert "warmed_at" in meta
    assert "torch_compile_cache_dir" in meta


def test_is_warm_returns_false_when_meta_deleted(_isolated_cache_root):
    """Deleting the meta.json sidecar flips ``is_warm`` back to False."""
    key = compile_cache.compute_key(**_BASE_KEY_KWARGS)
    compile_cache.mark_warm(key)
    assert compile_cache.is_warm(key) is True

    meta_path = compile_cache.cache_dir_for_key(key) / "meta.json"
    meta_path.unlink()
    assert compile_cache.is_warm(key) is False


# -- AC #3 row 6: is_warm() defends against directory-prefix collision ------- #


def test_is_warm_defends_against_directory_prefix_collision(_isolated_cache_root):
    """A 16-char prefix collision is detected as "not warm".

    The H1+H2 cache-invalidation discipline from
    ``memory/code_review_regression_test_exact_class.md`` mandates that the
    full-key match in ``meta.json`` (NOT the directory name) is the
    canonical "warm" signal. Two keys with the same 16-char prefix share
    the directory; ``is_warm`` must use the full key to disambiguate.

    Construction: write meta.json for ``key1``; query ``is_warm(key2)``
    where ``key2[:16] == key1[:16]`` but ``key1 != key2``. Expect False.
    """
    key1 = compile_cache.compute_key(**_BASE_KEY_KWARGS)
    # Fabricate a key2 with the same 16-char prefix but different full key.
    # SHA-256 prefix collisions are astronomically unlikely (2^-64), so
    # we construct key2 manually: take key1's prefix and pad with a
    # distinguishing suffix.
    key2 = key1[:16] + ("0" * (len(key1) - 16))
    # Sanity: prefixes match, full keys differ.
    assert key1[:16] == key2[:16]
    assert key1 != key2

    # Mark key1 warm; both keys point at the same directory.
    compile_cache.mark_warm(key1)
    assert compile_cache.cache_dir_for_key(key1) == compile_cache.cache_dir_for_key(key2)

    # is_warm(key1) is True (its meta matches); is_warm(key2) is False
    # because the meta.json's "key" field stores key1's full hash.
    assert compile_cache.is_warm(key1) is True
    assert compile_cache.is_warm(key2) is False


def test_is_warm_returns_false_for_unparseable_meta(_isolated_cache_root):
    """Defensive: an unparseable meta.json is treated as "not warm".

    Mirrors Story 17.2's ``_voice_clone_prompt_meta_is_valid`` discipline —
    a broken cache must not silently ship corrupt audio.
    """
    key = compile_cache.compute_key(**_BASE_KEY_KWARGS)
    meta_path = compile_cache.cache_dir_for_key(key) / "meta.json"
    # Write garbage that's not valid JSON.
    meta_path.write_text("not-json {{{{{", encoding="utf-8")

    assert compile_cache.is_warm(key) is False


# -- AC #3 row 7: set_torchinductor_cache_dir() idempotent + logs INFO once -- #


def test_set_torchinductor_cache_dir_is_idempotent(
    _isolated_cache_root,
    _snapshot_and_restore_torchinductor_env,
    _reset_set_torchinductor_logged_once,
    caplog,
):
    """Two calls leave ``os.environ["TORCHINDUCTOR_CACHE_DIR"]`` at the most-recent path; logs INFO once."""
    key1 = compile_cache.compute_key(**_BASE_KEY_KWARGS)
    key2 = compile_cache.compute_key(
        **{**_BASE_KEY_KWARGS, "precision_str": "fp32"}
    )
    assert key1 != key2  # Sanity: different keys → different directories

    with caplog.at_level(logging.INFO, logger="myvoice.services.tts_streaming.compile_cache"):
        compile_cache.set_torchinductor_cache_dir(key1)
        first_dir = os.environ["TORCHINDUCTOR_CACHE_DIR"]
        compile_cache.set_torchinductor_cache_dir(key2)
        second_dir = os.environ["TORCHINDUCTOR_CACHE_DIR"]

    # Second call overwrites the env var (idempotency contract: most-recent wins).
    assert first_dir == str(compile_cache.cache_dir_for_key(key1))
    assert second_dir == str(compile_cache.cache_dir_for_key(key2))
    assert first_dir != second_dir

    # Exactly one INFO log emission (set-once guard); the second call
    # logs at DEBUG (filtered out at INFO level).
    info_records = [
        r for r in caplog.records
        if r.levelno == logging.INFO
        and r.name == "myvoice.services.tts_streaming.compile_cache"
    ]
    assert len(info_records) == 1, (
        f"Expected exactly one INFO log; got {len(info_records)}: "
        f"{[r.getMessage() for r in info_records]}"
    )
