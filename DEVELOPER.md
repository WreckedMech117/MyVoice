# MyVoice V2 Developer Guide

A comprehensive guide for developers contributing to MyVoice.

## Quick Start

```bash
# Clone repository
git clone https://github.com/myvoice/myvoice.git
cd myvoice

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run application
python -m myvoice.main

# Run tests
pytest tests/
```

## Development Environment

### Prerequisites

- **Python**: 3.10+ (3.10 recommended for compatibility)
- **OS**: Windows 10/11 (primary target platform)
- **RAM**: 16GB recommended (models use ~3.4GB each)
- **IDE**: VS Code, PyCharm, or similar with Python support

### Dependencies

Core dependencies (see `requirements.txt`):
- **PyQt6** (>=6.6.0) - GUI framework
- **torch** (>=2.0.0) - ML framework for Qwen3-TTS
- **transformers** - Hugging Face model loading
- **openai-whisper** - Speech recognition
- **pyaudio** - Audio I/O
- **numpy** - Numerical processing

Development dependencies:
- **pytest** - Testing framework
- **pytest-qt** - PyQt testing utilities
- **pytest-asyncio** - Async test support

### Environment Setup

```bash
# Create isolated environment
python -m venv .venv
.venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Install development tools
pip install pytest pytest-qt pytest-asyncio black isort mypy

# Verify installation
python -c "import myvoice; print(myvoice.__version__)"
```

## Project Structure

```
MyVoice/
├── src/
│   └── myvoice/                    # Main application package
│       ├── __init__.py             # Package metadata, version
│       ├── main.py                 # Entry point
│       ├── app.py                  # Application controller (~125KB)
│       ├── models/                 # Data models (14 modules)
│       │   ├── audio_device.py     # AudioDevice, DeviceType enums
│       │   ├── voice_profile.py    # VoiceProfile, VoiceType enums
│       │   ├── emotion_profile.py  # EmotionPreset, EmotionProfile
│       │   ├── app_settings.py     # Application settings model
│       │   └── ...
│       ├── services/               # Business logic (28+ modules)
│       │   ├── core/               # Base service infrastructure
│       │   ├── integrations/       # External integrations
│       │   └── ...                 # See Service Architecture below
│       ├── ui/                     # User interface
│       │   ├── components/         # Reusable widgets
│       │   └── dialogs/            # Dialog windows
│       └── utils/                  # Utility functions
├── tests/                          # Test suite
│   ├── models/                     # Model tests
│   ├── services/                   # Service tests
│   └── ui/                         # UI tests
├── voice_files/                    # Voice data
│   ├── embeddings/                 # Saved voice embeddings
│   └── design_sessions/            # Voice Design working files
├── docs/                           # Documentation
│   └── stories/                    # User story files
├── build_tools/                    # Build and packaging
├── _bmad-output/                   # Planning artifacts
└── requirements.txt                # Python dependencies
```

## Service Architecture

MyVoice uses a service-oriented architecture with 28+ services organized into tiers.

### Base Service Pattern

All primary services inherit from `BaseService`:

```python
from myvoice.services.core.base_service import BaseService, ServiceStatus

class MyService(BaseService):
    def __init__(self):
        super().__init__("MyService")
        self.logger = logging.getLogger(__name__)

    async def start(self) -> bool:
        """Initialize and start the service."""
        await self._update_status(ServiceStatus.STARTING)
        # ... initialization logic ...
        self.status = ServiceStatus.RUNNING
        return True

    async def stop(self) -> bool:
        """Stop and cleanup the service."""
        # ... cleanup logic ...
        self.status = ServiceStatus.STOPPED
        return True

    async def health_check(self) -> tuple[bool, Optional[MyVoiceError]]:
        """Check service health."""
        return (True, None)  # or (False, error)
```

### Service Tiers

#### Tier 1: Core Services

| Service | Purpose |
|---------|---------|
| `AudioCoordinator` | Orchestrates dual-stream audio (monitor + virtual mic) |
| `MonitorAudioService` | Dedicated PyAudio for monitor speaker output |
| `VirtualMicrophoneService` | Dedicated PyAudio for virtual mic routing |
| `QwenTTSService` | Embedded Qwen3-TTS text-to-speech |
| `VoiceProfileManager` | Voice library and profile management |

#### Tier 2: Supporting Services

| Service | Purpose |
|---------|---------|
| `WhisperService` | OpenAI Whisper transcription |
| `ConfigurationManager` | Settings persistence (JSON) |
| `DeviceResilienceManager` | Device disconnect/reconnect handling |
| `TranscriptionQueueService` | Background transcription queue |
| `ModelLoadingManager` | Qwen3-TTS model lifecycle |

#### Tier 3: Utility Services

| Service | Purpose |
|---------|---------|
| `QuickSpeakService` | Quick phrase management |
| `TemplateService` | Voice template management |
| `AudioLevelManager` | Audio normalization |
| `DualStreamSynchronizer` | Stream synchronization |

### Service Composition

```
AudioCoordinator (Orchestrator)
├── MonitorAudioService
│   └── PyAudio (dedicated instance)
├── VirtualMicrophoneService
│   └── PyAudio (dedicated instance)
└── DeviceResilienceManager
    └── Device monitoring + callbacks
```

### Key Patterns

#### 1. Async/Threading Pattern
- Async methods for coordination and lifecycle
- Threading for blocking operations (audio playback)
- `ThreadPoolExecutor` for parallel processing
- `asyncio.Lock` for async critical sections
- `threading.Lock` for cross-thread synchronization

#### 2. Callback Pattern
```python
# Register callback
service.add_playback_complete_callback(my_handler)

# Emit callback
def _emit_playback_complete(self, task_id: str):
    for callback in self._playback_callbacks:
        callback(task_id)
```

#### 3. Configuration Pattern
```python
@dataclass
class ServiceConfig:
    sample_rate: int = 48000
    channels: int = 2
    chunk_size: int = 1024

service = MyService(config=ServiceConfig(sample_rate=44100))
```

#### 4. Error Handling Pattern
```python
class TTSErrorCode(Enum):
    OUT_OF_MEMORY = "out_of_memory"
    TIMEOUT = "timeout"
    EMPTY_TEXT = "empty_text"

# Return user-friendly errors
error = MyVoiceError(
    code=TTSErrorCode.TIMEOUT,
    user_message="Speech generation timed out. Try shorter text.",
    technical_details="Generation exceeded 30s limit"
)
```

## Data Models

Located in `src/myvoice/models/`:

| Model | Purpose |
|-------|---------|
| `AudioDevice` | Represents audio device (id, name, type) |
| `VoiceProfile` | Voice profile with type, path, metadata |
| `EmotionProfile` | Emotion preset or custom prompt |
| `AppSettings` | Application configuration |
| `TranscriptionResult` | Whisper transcription output |

### Voice Types

```python
class VoiceType(Enum):
    BUNDLED = "bundled"      # 9 pre-trained timbres (CustomVoice model)
    DESIGNED = "designed"    # Created from text description (VoiceDesign model)
    CLONED = "cloned"        # Cloned from audio sample (Base model)
```

### Emotion Support Matrix

| Voice Type | Emotion Control | Model Used |
|------------|-----------------|------------|
| BUNDLED | Yes | CustomVoice |
| DESIGNED | Yes | VoiceDesign |
| CLONED | No | Base |

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=myvoice tests/

# Run specific test file
pytest tests/services/test_audio_coordinator.py

# Run tests matching pattern
pytest -k "test_playback"

# Verbose output
pytest -v tests/
```

### Test Structure

```
tests/
├── conftest.py              # Shared fixtures
├── models/
│   └── test_voice_profile.py
├── services/
│   ├── test_audio_coordinator.py
│   ├── test_monitor_audio_service.py
│   └── test_device_resilience_manager.py
└── ui/
    └── test_main_window.py
```

### Writing Tests

```python
import pytest
from myvoice.services import AudioCoordinator

@pytest.fixture
def audio_coordinator():
    """Fixture providing AudioCoordinator instance."""
    coordinator = AudioCoordinator()
    yield coordinator
    # Cleanup after test
    asyncio.run(coordinator.stop())

@pytest.mark.asyncio
async def test_dual_stream_playback(audio_coordinator):
    """Test simultaneous monitor and virtual mic playback."""
    await audio_coordinator.start()

    result = await audio_coordinator.play_dual_stream(
        audio_data=test_audio,
        monitor_device=mock_monitor,
        virtual_device=mock_virtual
    )

    assert result.success
    assert result.monitor_played
    assert result.virtual_played
```

## UI Development

### Component Structure

```
ui/
├── components/
│   ├── emotion_button_group.py   # Emotion preset buttons
│   ├── voice_selector.py         # Voice dropdown
│   ├── quick_speak_menu.py       # Quick phrase menu
│   ├── status_bar.py             # Status indicators
│   └── device_notification.py    # Device warnings
└── dialogs/
    ├── settings_dialog.py        # Settings tabs
    ├── voice_design_dialog.py    # Voice Design UI
    └── voice_clone_dialog.py     # Voice Clone UI
```

### Creating New Components

```python
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PyQt6.QtCore import pyqtSignal

class MyComponent(QWidget):
    # Define signals for communication
    action_triggered = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        self.button = QPushButton("Action")
        layout.addWidget(self.button)

    def _connect_signals(self):
        self.button.clicked.connect(self._on_button_clicked)

    def _on_button_clicked(self):
        self.action_triggered.emit("action_name")
```

### Styling

Styles defined in QSS files. Follow existing patterns:

```css
/* Button styling */
QPushButton {
    background-color: #2d2d2d;
    border: 1px solid #3d3d3d;
    border-radius: 4px;
    padding: 8px 16px;
}

QPushButton:hover {
    background-color: #3d3d3d;
}

QPushButton:pressed {
    background-color: #1d1d1d;
}
```

## Adding New Features

### 1. Plan First
- Check `_bmad-output/planning-artifacts/` for existing requirements
- Review related stories in `docs/stories/`
- Identify affected services and models

### 2. Implement Service Logic
```python
# src/myvoice/services/my_new_service.py
class MyNewService(BaseService):
    def __init__(self):
        super().__init__("MyNewService")

    async def my_feature(self, params) -> Result:
        # Implementation
        pass
```

### 3. Add Tests
```python
# tests/services/test_my_new_service.py
@pytest.mark.asyncio
async def test_my_feature():
    service = MyNewService()
    result = await service.my_feature(test_params)
    assert result.success
```

### 4. Wire Up UI
```python
# Connect to app controller or existing UI component
self.my_service = MyNewService()
self.button.clicked.connect(self._on_feature_requested)
```

### 5. Update Documentation
- Add to this DEVELOPER.md if architectural
- Update README.md if user-facing
- Create story file if tracked feature

## Code Style

### General Guidelines
- Use type hints for function signatures
- Use dataclasses for data containers
- Prefer async/await over threading where possible
- Keep methods focused and small (<50 lines preferred)

### Naming Conventions
- Classes: `PascalCase`
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Import Order
```python
# Standard library
import asyncio
import logging
from pathlib import Path

# Third-party
from PyQt6.QtWidgets import QWidget
import numpy as np

# Local
from myvoice.services import AudioCoordinator
from myvoice.models import VoiceProfile
```

### Docstrings
```python
def process_audio(audio_data: bytes, sample_rate: int = 48000) -> np.ndarray:
    """Process raw audio data into numpy array.

    Args:
        audio_data: Raw audio bytes (WAV format)
        sample_rate: Target sample rate in Hz

    Returns:
        Normalized audio as numpy array

    Raises:
        ValueError: If audio_data is empty or invalid format
    """
```

## Common Tasks

### Running the Application

```bash
# Development mode
python -m myvoice.main

# With debug logging
python -m myvoice.main --debug

# From specific directory
cd src && python -m myvoice.main
```

### Building for Distribution

```bash
cd build_tools

# Build executable
pyinstaller myvoice.spec

# Build installer (requires Inno Setup)
"C:\Program Files (x86)\Inno Setup 6\ISCC.exe" installer.iss

# Full release build
build_release.bat
```

### Version Management

```bash
cd build_tools

# Show current version
python version.py

# Bump version
python version.py bump patch   # 2.0.0 -> 2.0.1
python version.py bump minor   # 2.0.0 -> 2.1.0
python version.py bump major   # 2.0.0 -> 3.0.0

# Update all version references
python version.py update-all
```

## Troubleshooting Development Issues

### PyAudio Installation Fails
```bash
# Install Visual C++ Build Tools first
# Then: pip install pyaudio
```

### Model Loading Issues
- Check disk space (~3.4GB per model)
- Verify internet connection for first download
- Check `logs/myvoice.log` for errors

### Tests Failing with Audio Errors
- Mock audio services in tests
- Use `pytest-qt` fixtures for UI tests
- Run audio tests on system with audio devices

### Import Errors
```bash
# Ensure package is installed in development mode
pip install -e .
```

## Resources

- **Project Board**: Check Archon MCP for tasks
- **Stories**: `docs/stories/` for feature requirements
- **Architecture**: `_bmad-output/planning-artifacts/architecture.md`
- **PRD**: `_bmad-output/planning-artifacts/prd.md`

---

*Last Updated: 2026-02-08*
*MyVoice V2.0.0*
