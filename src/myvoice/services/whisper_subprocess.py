"""
Subprocess-based Whisper transcription service.

This module provides a Whisper transcription service that runs in a separate process
to avoid DLL conflicts with PyQt6 on Windows.
"""

import json
import logging
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class TranscriptionResult:
    """Simple transcription result class to match whisper_service interface."""

    def __init__(self, text: str, language: str = "unknown", confidence: float = 0.9, duration: float = 0.0):
        self.text = text
        self.language = language
        self.confidence = confidence
        self.duration = duration


class WhisperSubprocessService:
    """
    Whisper transcription service using subprocess isolation.

    This service runs Whisper in a separate Python process to avoid
    DLL conflicts with PyQt6 on Windows systems.
    """

    def __init__(self):
        """Initialize the subprocess-based Whisper service."""
        self.logger = logging.getLogger(self.__class__.__name__)
        self.service_name = "WhisperService"

        # Import ServiceStatus for compatibility with @require_service_health decorator
        from myvoice.services.core.base_service import ServiceStatus
        self.status = ServiceStatus.STOPPED

    async def transcribe_file(self, file_path: Path, language: Optional[str] = None,
                            word_timestamps: bool = False, temperature: float = 0.0) -> TranscriptionResult:
        """
        Transcribe an audio file using Whisper.

        Args:
            file_path: Path to the audio file
            language: Language code (None for auto-detect)
            word_timestamps: Whether to include word-level timestamps
            temperature: Sampling temperature

        Returns:
            TranscriptionResult with transcription data
        """
        try:
            self.logger.info(f"Starting transcription for: {file_path}")

            # When frozen (PyInstaller), import Whisper directly - subprocess not needed
            # PyInstaller already isolated dependencies in the bundle
            if getattr(sys, 'frozen', False):
                self.logger.info("Running in PyInstaller bundle - using direct Whisper import")

                # Fix for Windows 11: Ensure stdout/stderr are valid file-like objects
                # PyInstaller GUI apps may have None stdout/stderr, causing tqdm to crash
                import io
                if sys.stdout is None:
                    sys.stdout = io.StringIO()
                    self.logger.info("Fixed None stdout for PyInstaller bundle")
                if sys.stderr is None:
                    sys.stderr = io.StringIO()
                    self.logger.info("Fixed None stderr for PyInstaller bundle")

                # Add bundled ffmpeg to PATH for Whisper audio processing
                import os
                app_dir = Path(sys._MEIPASS)
                ffmpeg_dir = app_dir / "ffmpeg"
                if ffmpeg_dir.exists():
                    os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
                    self.logger.info(f"Added bundled ffmpeg to PATH: {ffmpeg_dir}")
                else:
                    self.logger.warning(f"Bundled ffmpeg directory not found at {ffmpeg_dir}")

                import whisper
                import torch

                # Force CPU mode
                torch.cuda.is_available = lambda: False

                # Determine bundled model path
                app_dir = Path(sys._MEIPASS)
                model_dir = app_dir / "whisper_models"
                model_file = model_dir / "base.pt"

                self.logger.info(f"Loading bundled Whisper model from: {model_file}")

                if model_file.exists():
                    # Load bundled model directly
                    checkpoint = torch.load(str(model_file), map_location="cpu")

                    # Handle checkpoint format - may have model_state_dict wrapper
                    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                        # Checkpoint has wrapper with metadata
                        state_dict = checkpoint["model_state_dict"]
                    else:
                        # Direct state dict
                        state_dict = checkpoint

                    model = whisper.load_model("base", device="cpu", download_root=None)
                    model.load_state_dict(state_dict)
                    self.logger.info("Loaded bundled Whisper model successfully")
                else:
                    # Fallback: try to load from bundled directory
                    if model_dir.exists():
                        model = whisper.load_model("base", device="cpu", download_root=str(model_dir))
                    else:
                        raise RuntimeError(f"Bundled Whisper model not found at {model_file}")

                # Transcribe
                self.logger.info(f"Transcribing audio file: {file_path}")
                result = model.transcribe(
                    str(file_path),
                    language=language,
                    temperature=temperature,
                    word_timestamps=word_timestamps
                )

                # Create TranscriptionResult
                transcription_result = TranscriptionResult(
                    text=result["text"].strip(),
                    language=result.get("language", "unknown"),
                    confidence=0.9,  # Placeholder
                    duration=0.0  # Will be set by caller if needed
                )

                self.logger.info(f"Transcription completed: {len(transcription_result.text)} characters")
                return transcription_result

            else:
                # Running from source - use subprocess to avoid PyQt6 DLL conflicts
                self.logger.info("Running from source - using subprocess isolation")
                return await self._transcribe_subprocess(file_path, language, word_timestamps, temperature)

        except Exception as e:
            self.logger.error(f"Transcription failed: {e}", exc_info=True)
            raise

    async def _transcribe_subprocess(self, file_path: Path, language: Optional[str] = None,
                                   word_timestamps: bool = False, temperature: float = 0.0) -> TranscriptionResult:
        """Transcribe using subprocess (for non-frozen environments)."""
        try:
            # Create subprocess script
            script_content = self._create_whisper_script()

            # Write script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name

            try:
                # Use sys.executable for subprocess
                executable = sys.executable

                args = [
                    executable, script_path,
                    str(file_path),
                    '--temperature', str(temperature)
                ]

                if language:
                    args.extend(['--language', language])

                if word_timestamps:
                    args.append('--word-timestamps')

                # Run subprocess
                self.logger.debug(f"Running subprocess: {' '.join(args)}")
                result = subprocess.run(
                    args,
                    capture_output=True,
                    text=True,
                    timeout=300,  # 5 minute timeout
                    cwd=Path.cwd()
                )

                if result.returncode != 0:
                    error_msg = f"Subprocess failed: {result.stderr}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)

                # Parse result
                output = result.stdout.strip()
                if not output:
                    raise RuntimeError("No output from subprocess")

                result_dict = json.loads(output)

                # Create TranscriptionResult object
                transcription_result = TranscriptionResult(
                    text=result_dict.get('text', ''),
                    language=result_dict.get('language', 'unknown'),
                    confidence=result_dict.get('confidence', 0.9),
                    duration=result_dict.get('duration', 0.0)
                )

                self.logger.info(f"Transcription completed: {len(transcription_result.text)} characters")
                return transcription_result

            finally:
                # Clean up temporary script
                try:
                    Path(script_path).unlink()
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup script {script_path}: {e}")

        except Exception as e:
            self.logger.error(f"Subprocess transcription failed: {e}")
            raise

    def _create_whisper_script(self) -> str:
        """Create the Python script that will run Whisper in subprocess."""
        return '''
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Suppress Whisper warnings
logging.getLogger("whisper").setLevel(logging.ERROR)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Audio file to transcribe")
    parser.add_argument("--language", help="Language code")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--word-timestamps", action="store_true", help="Include word timestamps")

    args = parser.parse_args()

    try:
        import whisper
        import torch

        # Force CPU mode to avoid CUDA errors
        torch.cuda.is_available = lambda: False

        # Determine bundled model path and add ffmpeg to PATH
        # When frozen by PyInstaller, models are in whisper_models/ directory
        if getattr(sys, "frozen", False):
            # Running in PyInstaller bundle
            app_dir = Path(sys._MEIPASS)
            model_dir = app_dir / "whisper_models"

            # Add bundled ffmpeg to PATH
            ffmpeg_dir = app_dir / "ffmpeg"
            if ffmpeg_dir.exists():
                os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + os.environ.get("PATH", "")
        else:
            # Running from source - use project root
            app_dir = Path(__file__).parent.parent.parent.parent
            model_dir = app_dir / "whisper_models"

            # Add GPT-SoVITS ffmpeg to PATH if available
            gptsovits_ffmpeg = app_dir / "GPT-SoVITS"
            if gptsovits_ffmpeg.exists():
                os.environ["PATH"] = str(gptsovits_ffmpeg) + os.pathsep + os.environ.get("PATH", "")

        # Check if bundled model exists
        model_file = model_dir / "base.pt"

        if model_file.exists():
            # Load bundled model directly
            import torch
            checkpoint = torch.load(str(model_file), map_location="cpu")

            # Handle checkpoint format - may have model_state_dict wrapper
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # Checkpoint has wrapper with metadata
                state_dict = checkpoint["model_state_dict"]
            else:
                # Direct state dict
                state_dict = checkpoint

            model = whisper.load_model("base", device="cpu", download_root=None)
            model.load_state_dict(state_dict)
        else:
            # Fallback: try to load from bundled directory or download
            if model_dir.exists():
                model = whisper.load_model("base", device="cpu", download_root=str(model_dir))
            else:
                # Last resort: use default cache (should not happen in production)
                model = whisper.load_model("base", device="cpu")

        # Transcribe
        result = model.transcribe(
            args.file_path,
            language=args.language,
            temperature=args.temperature,
            word_timestamps=args.word_timestamps
        )

        # Create simplified result
        output = {
            "text": result["text"].strip(),
            "language": result.get("language", "unknown"),
            "confidence": 0.9,  # Placeholder
            "duration": 0.0  # Will be set by caller
        }

        # Output JSON result
        print(json.dumps(output))

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
'''

    async def start(self):
        """Start the service (no-op for subprocess version)."""
        from myvoice.services.core.base_service import ServiceStatus
        self.status = ServiceStatus.RUNNING
        self.logger.info("Subprocess Whisper service ready")
        return True

    async def stop(self):
        """Stop the service (no-op for subprocess version)."""
        from myvoice.services.core.base_service import ServiceStatus
        self.status = ServiceStatus.STOPPED
        self.logger.info("Subprocess Whisper service stopped")
        return True

    def is_running(self) -> bool:
        """Check if service is running."""
        return True