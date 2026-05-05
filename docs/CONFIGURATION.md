# MyVoice Configuration Reference

This document describes all configuration files, settings, and data structures used by MyVoice V2.

## File Locations

| File/Directory | Location | Purpose |
|----------------|----------|---------|
| `settings.json` | `%LOCALAPPDATA%\MyVoice\` | Application settings |
| `voice_files/` | Application directory | Voice samples and embeddings |
| `voice_files/embeddings/` | Application directory | Saved voice embeddings |
| `voice_files/design_sessions/` | Application directory | Temporary Voice Design files |
| `config/quickspeak_profiles/` | Application directory | Quick Speak phrase profiles |
| `logs/` | Application directory | Application logs |

## Settings File (settings.json)

The main configuration file located at `%LOCALAPPDATA%\MyVoice\settings.json`.

### Full Schema

```json
{
  "_settings_version": "1.0",
  "_last_modified": 1707350400.0,

  "selected_voice_profile": "Sarira-F",
  "voice_files_directory": "voice_files",
  "recent_voice_profiles": ["Sarira-F", "Ryan", "MyClonedVoice"],
  "max_voice_duration": 10.0,
  "auto_refresh_interval": 30,
  "config_directory": "config",

  "log_level": "INFO",

  "ui_theme": "dark",
  "always_on_top": true,
  "window_geometry": {
    "x": 100,
    "y": 100,
    "width": 400,
    "height": 188
  },
  "window_transparency": 1.0,
  "minimize_to_tray": true,
  "tray_notification_shown": false,

  "tts_service_url": "http://localhost:9880",
  "tts_service_timeout": 30,

  "enable_audio_monitoring": true,
  "monitor_device_id": "device-guid-here",
  "monitor_device_name": "Speakers (Realtek Audio)",
  "monitor_device_host_api": "Windows WASAPI",
  "virtual_microphone_device_id": "device-guid-here",
  "virtual_microphone_device_name": "CABLE Input (VB-Audio)",
  "virtual_microphone_device_host_api": "Windows WASAPI",

  "training_enabled": true,
  "training_workspace_directory": "training_workspace",

  "model_tier": "quality",

  "custom_emotion_text": null,
  "custom_emotion_presets": [
    "Rising Frustration",
    "Growing Excitement",
    "Trailing Off Sadly",
    "Building Confidence",
    "Hesitant and Uncertain",
    "Warm and Reassuring",
    "Cold and Distant",
    "Playful Teasing",
    "Sincere Apology",
    "Dramatic Emphasis"
  ],

  "advanced_settings": {}
}
```

### Setting Descriptions

#### Voice Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `selected_voice_profile` | string | `"Sarira-F"` | Currently selected voice profile name |
| `voice_files_directory` | string | `"voice_files"` | Directory for voice samples |
| `recent_voice_profiles` | array | `[]` | Recently used voice profiles (max 10) |
| `max_voice_duration` | float | `10.0` | Maximum voice sample duration in seconds |
| `auto_refresh_interval` | int | `30` | Voice library refresh interval in seconds |

#### UI Settings

| Setting | Type | Default | Valid Values | Description |
|---------|------|---------|--------------|-------------|
| `ui_theme` | string | `"dark"` | `"dark"`, `"light"` | Application theme |
| `always_on_top` | bool | `true` | `true`, `false` | Keep window above other apps |
| `window_geometry` | object | `null` | See below | Window position and size |
| `window_transparency` | float | `1.0` | `0.2` - `1.0` | Window opacity (20%-100%) |
| `minimize_to_tray` | bool | `true` | `true`, `false` | Minimize to system tray |
| `tray_notification_shown` | bool | `false` | `true`, `false` | One-time tray notification shown |

#### Window Geometry Object

```json
{
  "x": 100,      // Window X position (pixels)
  "y": 100,      // Window Y position (pixels)
  "width": 400,  // Window width (320-600)
  "height": 188  // Window height (168-268)
}
```

#### Audio Settings

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `enable_audio_monitoring` | bool | `true` | Enable audio output monitoring |
| `monitor_device_id` | string | `null` | Monitor speaker device GUID |
| `monitor_device_name` | string | `null` | Monitor speaker friendly name |
| `monitor_device_host_api` | string | `null` | Audio host API (e.g., "Windows WASAPI") |
| `virtual_microphone_device_id` | string | `null` | Virtual mic device GUID |
| `virtual_microphone_device_name` | string | `null` | Virtual mic friendly name |
| `virtual_microphone_device_host_api` | string | `null` | Virtual mic host API |

#### Logging

| Setting | Type | Default | Valid Values |
|---------|------|---------|--------------|
| `log_level` | string | `"INFO"` | `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"` |

#### Model Tier

| Setting | Type | Default | Valid Values | Description |
|---------|------|---------|--------------|-------------|
| `model_tier` | string | `"quality"` | `"quality"`, `"small"` | TTS model tier selection |

- **quality**: Uses 1.7B parameter models (~3.4 GB each). Higher quality output.
- **small**: Uses 0.6B parameter models (~1.2 GB each). Faster inference, lower memory.

**Note:** Embedding dimensions differ between tiers (2048 vs 1024). Voices with tier-specific embeddings will fall back to source audio when using the opposite tier.

#### Custom Emotions

| Setting | Type | Default | Description |
|---------|------|---------|-------------|
| `custom_emotion_text` | string | `null` | Current custom emotion prompt text |
| `custom_emotion_presets` | array | See below | User-defined emotion presets |

Default custom emotion presets:
- Rising Frustration
- Growing Excitement
- Trailing Off Sadly
- Building Confidence
- Hesitant and Uncertain
- Warm and Reassuring
- Cold and Distant
- Playful Teasing
- Sincere Apology
- Dramatic Emphasis

## Voice Profile Structure

### Voice Types

| Type | Icon | Model Used | Emotion Support | Storage |
|------|------|------------|-----------------|---------|
| `bundled` | 📦 | CustomVoice | Yes | Virtual path |
| `designed` | 🎭 | VoiceDesign | Yes | `voice_files/designed/` |
| `cloned` | 🎤 | Base | No | `voice_files/cloned/` |
| `embedding` | 🧬 | Base | Yes (multi-embedding) | `voice_files/embeddings/` |
| `optimized` | 🔧 | CustomVoice | Yes | Fine-tuned checkpoint |

### Embedding Voice Directory Structure

```
voice_files/embeddings/{voice_name}/
├── metadata.json           # Voice metadata (v3.0 schema)
├── preview.wav             # Preview audio file
├── neutral/
│   ├── 1.7/
│   │   └── embedding.pt    # Quality tier (2048-dim)
│   ├── 0.6/
│   │   └── embedding.pt    # Small tier (1024-dim)
│   └── source_audio.wav    # Shared source for fallback
├── happy/
│   ├── 1.7/
│   │   └── embedding.pt
│   ├── 0.6/
│   │   └── embedding.pt
│   └── source_audio.wav
├── sad/
│   ├── 1.7/embedding.pt
│   └── 0.6/embedding.pt
├── angry/
│   ├── 1.7/embedding.pt
│   └── 0.6/embedding.pt
└── flirtatious/
    ├── 1.7/embedding.pt
    └── 0.6/embedding.pt
```

**Embedding Resolution Order:**
1. `{emotion}/{tier}/embedding.pt` (tier-specific)
2. `{emotion}/embedding.pt` (legacy v2.0 fallback)
3. Falls back to `source_audio.wav` if tier mismatch detected

### Embedding Metadata (metadata.json)

```json
{
  "version": "3.0",
  "name": "My Custom Voice",
  "description": "Warm, friendly voice with slight accent",
  "available_emotions": ["neutral", "happy", "sad", "angry", "flirtatious"],
  "available_tiers": {
    "neutral": ["1.7", "0.6"],
    "happy": ["1.7", "0.6"],
    "sad": ["1.7"],
    "angry": ["1.7"],
    "flirtatious": ["0.6"]
  },
  "created_at": "2026-02-01T12:00:00",
  "updated_at": "2026-02-01T12:00:00"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `version` | string | Schema version (`"1.0"`, `"2.0"`, or `"3.0"`) |
| `name` | string | Display name for the voice |
| `description` | string | Voice description text |
| `available_emotions` | array | List of available emotions |
| `available_tiers` | object | Map of emotion → available tier IDs (v3.0+) |
| `created_at` | string | ISO 8601 creation timestamp |
| `updated_at` | string | ISO 8601 last modified timestamp |

**Version History:**
- **v1.0**: Basic metadata
- **v2.0**: Added emotion directories with embeddings
- **v3.0**: Added multi-tier support with `available_tiers` field

### Valid Emotions

The following emotions are supported:
- `neutral` - Default, natural tone
- `happy` - Upbeat, cheerful
- `sad` - Somber, melancholic
- `angry` - Intense, forceful
- `flirtatious` - Playful, teasing

## Quick Speak Profiles

Quick Speak phrases are stored in CSV files in `config/quickspeak_profiles/`.

### CSV Format

```csv
id,text,label
1,"Hello, how are you?","Greeting"
2,"Thank you very much!","Thanks"
3,"I'll be right back.",""
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique entry ID |
| `text` | string | Phrase text to speak |
| `label` | string | Optional short label (for display) |

### Default Profile

The default profile (`default.csv`) is created automatically with sample phrases on first launch.

## Configuration Recovery

### Corrupted Settings

If `settings.json` becomes corrupted:
1. Application backs up corrupted file as `settings.json.bak`
2. Creates new settings with defaults
3. Logs warning about recovery

### Missing Voice

If selected voice is deleted:
1. Falls back to default bundled voice (`Sarira-F`)
2. If default missing, selects first available voice
3. Updates settings with new selection

### Backup Files

Up to 3 backup files are maintained:
- `settings.json.bak` - Most recent backup
- `settings.json.bak.1` - Previous backup
- `settings.json.bak.2` - Oldest backup

## Example Configurations

### Minimal Configuration

```json
{
  "_settings_version": "1.0",
  "selected_voice_profile": "Ryan",
  "ui_theme": "dark",
  "always_on_top": true
}
```

### Gaming Setup (Low Latency)

```json
{
  "selected_voice_profile": "Aiden",
  "always_on_top": true,
  "window_transparency": 0.8,
  "minimize_to_tray": true,
  "virtual_microphone_device_name": "CABLE Input (VB-Audio)"
}
```

### Accessibility Configuration

```json
{
  "ui_theme": "light",
  "always_on_top": true,
  "window_transparency": 1.0,
  "log_level": "DEBUG"
}
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_HOME` | Custom HuggingFace model cache location |
| `CUDA_VISIBLE_DEVICES` | GPU selection (empty = CPU only) |
| `MYVOICE_DEBUG` | Enable debug logging (`1` = enabled) |

## Validation Rules

### Window Transparency
- Range: 0.0 to 1.0
- Values below 0.2 are clamped to 0.2 (20% minimum)
- Invalid values reset to 1.0 (fully opaque)

### Log Level
- Must be one of: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- Invalid values reset to `INFO`

### Voice Duration
- Must be positive (> 0)
- Warning if > 60 seconds
- Recommended: 3-10 seconds for cloning

---

*Last Updated: 2026-02-20*
*MyVoice V2.5.0*
