# Story tooling-5 — Evidence: Clear the Pre-Existing Failure Backlog

Date: 2026-09-14. Branch `tooling-5-failure-backlog` off clean `main`
(`8327f06`). Interpreter: bundled `python310\python.exe` (3.10.11), pytest
9.0.2, pytest-timeout 2.4.0 (60 s, thread), PyQt6 6.9.1. tooling-4's
raise-on-modal net is active throughout.

Baseline re-confirmed on this branch before any edit: `pytest tests/` →
**49 failed, 2924 passed, 5 errors in 83.99 s**, ids identical to
`tooling-4/suite-T4-run1.{failed,errors}`. Raw log of the closing run:
`tooling-5/suite-T5-run1.log`.

## 0. Result in one paragraph

A single `pytest tests/ -rfEx` now reports **2969 passed, 1 xfailed, 0 failed,
0 errors in 80.31 s**. The 54 rows fell into **16 root causes** (§1). Fifteen
are stale tests — the product changed on purpose and the test was not moved;
each assertion was rewritten to the current, intended behaviour with a comment
naming the story/QA round that changed it. One is a real product defect
(§2): the QA8 relocation of the clone flow dropped the "loading a sample marks
unsaved work" wiring. Whether a merely-selected sample should count as work is
a product call, so it is `xfail(strict=True)` with the defect in the reason
rather than fixed or hidden. **`src/` is untouched** (`git diff --stat`: 13
files, all under `tests/`). `quick_speak_settings_widget.py`, its tests, and
`app.py` were not opened for editing.

## 1. AC #1 — The 54 by root cause

| # | Root cause | Rows | Verdict | What changed |
|---|---|---|---|---|
| 1 | **QA3 split `SamplePathPanel` into Sample/Clone sub-tabs.** Transcript, extract, preview and modified-indicator widgets live on the Clone page (index 1), which is not current after construction, so Qt's `isVisible()` is False for them even though the panel is shown. The 11 positive-visibility tests never switched pages. | 11 (`test_sample_path_panel.py`) | stale | One helper `_open_clone_sub_tab(panel)` (`sub_tabs.setCurrentIndex(1)`, what the QA4 Next button does) inserted after the file load in each of the 11 tests. No `isVisible` → `isHidden` softening. |
| 2 | **QA8 removed the "From Sample" tab** (cloning moved to From Description > Clone sub-tab) and the Emotion Variants feature added Emotions + Refinement, so the dialog has 3 tabs, no `sample_panel`, accessible name "Voice creation workflow". | 6 (`test_has_two_tabs`, `test_second_tab_is_from_sample`, `test_tab_widget_has_accessible_name`, `TestSampleTab::test_sample_panel_exists`, `::test_sample_panel_is_sample_path_panel`, `::test_sample_tab_has_browse_button`) | stale | Tabs asserted as From Description / Emotions / Refinement; Clone pinned as `description_panel.sub_tabs` index 2; `sample_panel` asserted absent; browse button asserted on `description_panel.clone_browse_button`; clone request handlers asserted wired. |
| 3 | **QA5 removed the footer Save button**; `_on_save_ready_changed` is an explicit no-op; footer is New Voice + Cancel. | 11 (`test_save_button_exists`, `test_save_button_initially_disabled`, `test_save_button_has_accessible_name`, `TestSaveButtonState` ×5, `TestSampleTab::test_save_button_disabled_on_sample_tab_initially`, `TestSampleTabSaveReady` ×2) | stale | New Voice button asserted (existence, accessible name); `save_button` asserted absent; `TestSaveButtonState` collapsed to 2 tests (signal stays wired and is harmless; ready panel survives a tab round-trip). `TestSampleTabSaveReady` (2) dropped — they drove the removed Save button from the removed sample panel; no counterpart exists. |
| 4 | **QA4 made `SessionManager` a single persistent `design_sessions/current`** (variants survive close/reopen); UUID sessions only behind `use_persistent=False`. | 5 (4 in `test_session_manager.py`, `test_session_dir_is_unique` in the dialog file) | stale | Default asserted as `PERSISTENT_SESSION_NAME == "current"` and shared across instances; the uniqueness / cleanup-isolation tests moved to `use_persistent=False`; two tests added so both modes are pinned. |
| 5 | **`AudioPlayerWidget.set_audio_file` hands the path to `QMediaPlayer.setSource`**; the Windows media backend holds the file until the source is cleared. The `temp_audio_file` fixture unlinked eagerly, and fixture teardown runs in reverse setup order, so the unlink ran while the `widget` fixture still held the file (`deleteLater` never fires under pytest). | 5 errors (`test_audio_player_widget.py`) | stale test hygiene | `widget` fixture teardown calls `widget.clear()` (releases the source, the production release path); `temp_audio_file` writes under `tmp_path` instead of `NamedTemporaryFile` + unlink. Observation for the record: `SessionManager.clear_variant_files` catches per-file exceptions, so the same handle hold in production degrades to a logged warning, not a crash — not in this story's scope. |
| 6 | **V2 Qwen3-TTS migration (7d36963) rewrote the emotion instructs**: UI presets became acted direction ("joyful enthusiasm … pure happiness", "deep sorrow", "rage", "teasing allure"), service presets became adverbs ("Speak happily"). Both tests did `emotion in instruct.lower()`; "happy" is not a substring of "happily"/"happiness", and the UI sad/angry/flirtatious strings paraphrase entirely. | 2 (`test_emotion_tts_integration.py`) | stale | Module-level `EMOTION_INSTRUCT_STEMS` (`happ/joy/cheer`, `sad/sorrow/melanchol`, `angr/rage`, `flirt/teas`); each instruct must carry one of its emotion's stems, checked on both sides in the cross-check test. |
| 7 | **Embedding metadata schema is v3.0** (multi-tier `available_tiers`, `docs/CONFIGURATION.md`); `to_embedding_metadata` never writes 2.0. | 1 (`test_generate_v2_metadata`) | stale | Renamed `test_generate_v3_metadata`; asserts `"3.0"` and `available_tiers` present. v2.0 *parsing* is still covered by `test_v2_metadata_full`. |
| 8 | **Story 18.1 code-review M1 added `AudioChunk.session_id`.** | 1 (`test_audio_chunk_field_set_unchanged`) | stale | Baseline set now includes `session_id` with the story reference; the set is still pinned exactly. |
| 9 | **Story 20.5 Phase 4 reads `getattr(self, "_tts_service", None)` in the device path**; on the test's `__new__`'d `MyVoiceApp` PyQt's `QObject.__getattr__` raises `RuntimeError` instead of returning the default, the handler's `except Exception` logged "falling back to batch" and never opened the streaming session. The test's own docstring already documents this hazard for other attrs. | 1 (`test_gui_session_reaches_device_and_is_not_published`) | stale | `_make_app()` pre-sets `app._tts_service = None`. |
| 10 | **V2 added `VoiceType.EMBEDDING` at sort_order 1**, shifting OPTIMIZED from 2 to 3. | 1 (`test_optimized_sort_order`) | stale | Asserts 3 and the relative order DESIGNED < OPTIMIZED < CLONED the test was about. |
| 11 | **`ModelRegistry._load_model` gained `tier_override`** (multi-tier quality setting); `ensure_model_loaded` forwards it unconditionally, so the test's `fake_load(model_type, checkpoint_path=None)` got an unexpected kwarg. | 1 (`test_checkpoint_path_tracked`) | stale | `fake_load` accepts `tier_override=None`. |
| 12 | **V2 raised `VoiceProfileManager.max_duration` default 10 → 300 s** (Qwen3-TTS handles long reference files). | 1 (`test_init_with_defaults`) | stale | Asserts 300.0. |
| 13 | **`AppSettings.minimize_to_tray` defaults to False**, pinned deliberately by Story ui-1 (`test_close_to_tray_toggle.py::test_default_setting_value_is_unchanged`); the tray test asserted the Story 7.2 draft value True. | 1 (`test_minimize_to_tray_default_is_true`) | stale | Renamed `..._is_false`, asserts `is False`, cites ui-1. |
| 14 | **V2 removed the library's Clone Voice button/signal** (cloning is inside the studio) and **gated Design Voice on `set_tts_available`** — a disabled `QPushButton` swallows `click()`. | 4 (`test_voice_library_widget.py`) | stale | Clone button/signal asserted absent; new test pins the TTS-availability gate; `test_design_button_emits_signal` asserts no emission while disabled, then one after `set_tts_available(True)`. |
| 15 | **Story 15.3 asserted Clear Comms is the *last* settings tab**; Story 16.6 inserted Streaming before it (keeping it last), then the local TTS API (tech-spec local-tts-api v1, `3e3e740`) appended API Access after it. | 2 (`test_settings_dialog_clear_comms_tab.py`) | stale (per story note: "the API Access tab was added after Clear Comms") | Tab located by label; asserts it follows Audio, Voices, TTS, Interface, Quick Speak, Streaming (index 6) and holds the `ClearCommsSettingsPanel`. |
| 16 | **Defect — QA8 dropped the unsaved-work wiring for the clone flow.** Story 2.1's `SamplePathPanel` load set `dialog._has_unsaved_work`; `DescriptionPathPanel._load_clone_audio_file` emits `clone_file_loaded`, but `VoiceDesignStudioDialog` never connects it to `set_has_unsaved_work`, so `dialog_closing` and the New Voice confirmation treat a loaded sample as no work. | 1 (`test_file_loaded_sets_unsaved_work`) | **product wrong** | Rewritten against the Clone sub-tab (`test_clone_file_loaded_sets_unsaved_work`) and marked `xfail(strict=True)` naming the defect. Verified with `--runxfail` that it fails on exactly `assert dialog._has_unsaved_work is True`. |

Row count: 11 + 6 + 11 + 5 + 5 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 4 + 2 + 1 = 54.

## 2. AC #2 — The one real defect, and why it is xfail not fixed

The fix is one line in `voice_design_studio_dialog.py`
(`self.description_panel.clone_file_loaded.connect(lambda _p: self.set_has_unsaved_work(True))`),
so it is *small*. It is not *unambiguous*: `_has_unsaved_work` also drives the
"Start New Voice — this will clear your current session" confirmation, so the
fix would make picking a file (no generation, nothing in the session dir)
prompt on New Voice. Whether an uploaded-but-unprocessed sample is "work" is
a UX decision, and the alternative reading (only generated/extracted artefacts
count) is defensible. Listed for a product story; the strict xfail will XPASS
and fail the suite the moment someone wires it, so it cannot silently go
stale.

No assertion anywhere was changed to match a bug. Every "stale" verdict above
points at a source-level comment (`QA3`/`QA4`/`QA5`/`QA8`), a story file
(18.1, 20.5, ui-1, 15.3, 16.6), a doc (`CONFIGURATION.md` schema v3.0) or the
V2 migration commit (`7d36963`) that made the change on purpose.

## 3. AC #3 — Single run

```
pytest tests/ -p no:cacheprovider -rfEx
2969 passed, 1 xfailed in 80.31s (0:01:20)
```

0 failed, 0 errors, 0 timeouts, 0 raise-on-modal hits. The one xfail carries
the defect in its reason (visible in the `-rx` summary line at the end of
`tooling-5/suite-T5-run1.log`). Collected count moved 2973 → 2970: the dialog
file is net −5 (−3 `TestSaveButtonState`, −1 `TestSampleTab`, −2
`TestSampleTabSaveReady`, +1 `test_clone_lives_in_description_sub_tab`),
`test_session_manager.py` is +2, `test_voice_library_widget.py` is ±0.

## 4. AC #4 — `src/` diffs

None. `git diff --stat` lists 13 files, all under `tests/`:

```
tests/integration/test_emotion_tts_integration.py
tests/integration/test_emotion_variants_workflow.py
tests/integration/test_session_lifecycle.py
tests/services/api_server/test_origin_gating.py
tests/services/test_optimized_voice.py
tests/services/test_voice_profile_manager.py
tests/ui/test_settings_dialog_clear_comms_tab.py
tests/ui/test_system_tray_integration.py
tests/ui/test_voice_library_widget.py
tests/unit/ui/dialogs/voice_design_studio/test_audio_player_widget.py
tests/unit/ui/dialogs/voice_design_studio/test_sample_path_panel.py
tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py
tests/utils/test_session_manager.py
```

## 5. Side findings (not acted on)

- `SamplePathPanel` (`sample_path_panel.py`, 1546 lines) is no longer
  instantiated by any production code after QA8 — only re-exported from the
  package `__init__`. Its 137 tests now pass but exercise dead code. Candidate
  for a cleanup story.
- `AudioPlayerWidget` keeps its media source open until `clear()`; on Windows
  that blocks deletion of the file it points at. `clear_variant_files` tolerates
  this per-file, so Regenerate would log a warning and leave a stale variant
  behind rather than fail. Worth a look in whichever story next touches the
  studio's variant lifecycle.
