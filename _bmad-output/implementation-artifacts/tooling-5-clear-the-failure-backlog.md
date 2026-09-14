# Story tooling-5: Clear the Pre-Existing Failure Backlog

Status: in-progress

<!-- Out-of-epic tooling story. Source: the 49 failures + 5 errors documented as "pre-existing" since Story 20.2 and carried unchanged through tooling-4. -->
<!-- Risk: LOW-MEDIUM. Mostly stale tests. The risk is papering over a real product bug by "fixing" its test — AC #2 exists for that. -->

## Story

As **anyone reading a regression result on this repo**,
I want **a failing test to mean something is wrong**,
so that **"49 pre-existing failures, unchanged" stops being a sentence anyone has to write**.

## Context

tooling-4's single run: 49 failed, 5 errors, identical in identity to the set
first documented in Story 20.2's evidence. By file:

    19  tests/unit/ui/dialogs/voice_design_studio/test_voice_design_studio_dialog.py
    11  tests/unit/ui/dialogs/voice_design_studio/test_sample_path_panel.py
     5  tests/unit/ui/dialogs/voice_design_studio/test_audio_player_widget.py (errors)
     4  tests/utils/test_session_manager.py
     4  tests/ui/test_voice_library_widget.py
     2  tests/ui/test_settings_dialog_clear_comms_tab.py  (API Access tab added after Clear Comms)
     2  tests/services/test_optimized_voice.py
     2  tests/integration/test_emotion_tts_integration.py
     1  tests/ui/test_system_tray_integration.py  (asserts a True default QA3-4 changed to False)
     1  tests/services/test_voice_profile_manager.py
     1  tests/services/api_server/test_origin_gating.py
     1  tests/integration/test_session_lifecycle.py  (AudioChunk gained session_id in Story 18.1)
     1  tests/integration/test_emotion_variants_workflow.py

35 of 54 are in three voice-design-studio files — expect one or two root causes.
Ids: `_bmad-output/implementation-artifacts/tooling-4/suite-T4-run1.{failed,errors}`.

## Acceptance Criteria

### AC #1 — Every failure is root-caused before it is touched
**Then** the evidence file groups the 54 by root cause, not by file, and states
for each group whether the test is stale (product changed, test did not) or the
product is wrong

### AC #2 — A stale test is fixed; a real bug is reported, not hidden
**Given** a group where the test is stale
**Then** the assertion is updated to the current, intended behaviour, with a
comment naming the story that changed it
**Given** a group where the product is actually wrong
**Then** fix it only if the fix is small and unambiguous; otherwise leave the
test failing, mark it `xfail(strict=True, reason=...)` naming the defect, and
list it for a product story. Never change an assertion to match a bug

### AC #3 — The suite ends green, or every remaining red is an intentional xfail
**Then** a single `pytest tests/` reports 0 unexpected failures and 0 errors,
and every `xfail` carries a reason pointing at a named defect

### AC #4 — Nothing outside `tests/` changes unless AC #2's "small and unambiguous" applies
**Then** `src/` diffs, if any, are listed individually with the failing test
that justified each

## Dev Notes
ui-2 owns `quick_speak_settings_widget.py` and its tests; 20.9 owns `app.py`.
Do not touch those. The two Clear-Comms-tab assertions ARE yours.
