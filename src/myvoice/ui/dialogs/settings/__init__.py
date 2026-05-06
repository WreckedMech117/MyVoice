"""Clear Comms settings package (Story 15.1+, Phase 5 of D-20).

Re-exports the PRELOADED audio source loader added in Story 15.1.
Story 15.3 will add the ``ClearCommsSettingsPanel`` ``QWidget``
class to ``clear_comms_settings_panel.py``.
"""

from .clear_comms_settings_panel import (
    PreloadedAudioLoadError,
    WAV_FILE_DIALOG_FILTER,
    load_preloaded_audio_source,
)

__all__ = [
    "PreloadedAudioLoadError",
    "WAV_FILE_DIALOG_FILTER",
    "load_preloaded_audio_source",
]
