"""Epic 19 — the installer's Lightning choice on the model-selection page."""
import json
import re
from pathlib import Path

from myvoice.models.app_settings import AppSettings

REPO = Path(__file__).resolve().parents[2]
ISS = (REPO / "build_tools" / "installer.iss").read_text(encoding="utf-8")
DEFAULT_VOICES = REPO / "src" / "install_files" / "default_voices"


def _default_voice() -> str:
    return re.search(r'#define LightningDefaultVoice "([^"]+)"', ISS).group(1)


def test_model_page_offers_three_exclusive_choices():
    tasks = re.findall(r'^Name: "(modelquality_\w+)";.*Flags: exclusive', ISS, re.M)
    assert tasks == ["modelquality_quality", "modelquality_small", "modelquality_lightning"]


def test_lightning_choice_installs_and_downloads_the_small_tier():
    """Bundled voices, voice-from-description and new-voice creation still use
    the standard engine, so the Small models must be there."""
    both = "WizardIsTaskSelected('modelquality_small') or WizardIsTaskSelected('modelquality_lightning')"
    assert ISS.count(both) == 2  # settings writer + model pre-download


def test_lightning_default_voice_is_a_shipped_cloned_voice_with_everything_it_needs():
    """Lightning speaks cloned voices only; a fresh install must start on one
    that has its clip, transcript and Small-tier prompt cache (the reference
    codes come from that cache)."""
    voice = _default_voice()
    for suffix in (".wav", ".txt", ".small.pt"):
        assert (DEFAULT_VOICES / f"{voice}{suffix}").is_file(), suffix


def test_settings_the_installer_writes_turn_lightning_on_at_first_launch():
    block = re.search(
        r"if WizardIsTaskSelected\('modelquality_lightning'\) then.*?SettingsContent := (.*?)\n\s*else",
        ISS, re.S,
    ).group(1)
    text = "".join(re.findall(r"'((?:[^']|'')*)'", block))
    text = text.replace("{#LightningDefaultVoice}", _default_voice())
    text = text.replace("' + QualityTier + '", "small").replace("QualityTier", "small")
    written = json.loads(re.sub(r'"model_quality_tier": "[^"]*"', '"model_quality_tier": "small"', text))

    settings = AppSettings.from_dict(written)
    assert settings.lightning_mode is True
    assert settings.model_quality_tier == "small"
    assert settings.selected_voice_profile == _default_voice()
