"""Local TTS API (OpenAI-compatible) server package.

A Settings-toggled, localhost-bound HTTP server exposing an OpenAI
``/v1/audio/speech``-compatible API over MyVoice's existing QwenTTSService.
Runs as a coroutine on the app's single qasync event loop.

See: _bmad-output/implementation-artifacts/tech-spec-local-tts-api-v1.md
"""
