# Qwen3-TTS Integration Guide

MyVoice V2 uses Qwen3-TTS as the embedded text-to-speech engine. This document consolidates all technical details for the integration.

## Overview

Qwen3-TTS is a family of speech synthesis models developed by Qwen Team. MyVoice supports two model tiers:

- **Quality Tier (1.7B)**: Higher quality output, requires ~3.4 GB per model
- **Small Tier (0.6B)**: Faster inference, lower memory, requires ~1.2 GB per model

## Model Tiers

### Quality Tier (1.7B Parameters)

| Model | HuggingFace ID | Size | Purpose | Emotion Support |
|-------|----------------|------|---------|-----------------|
| **Base** | `Qwen/Qwen3-TTS-1.7B` | ~3.4 GB | Voice cloning from audio samples | No |
| **CustomVoice** | `Qwen/Qwen3-TTS-1.7B-CustomVoice` | ~3.4 GB | Pre-trained bundled timbres | Yes |
| **VoiceDesign** | `Qwen/Qwen3-TTS-1.7B-VoiceDesign` | ~3.4 GB | Text-description voice generation | Yes |

### Small Tier (0.6B Parameters)

| Model | HuggingFace ID | Size | Purpose | Emotion Support |
|-------|----------------|------|---------|-----------------|
| **Base** | `Qwen/Qwen3-TTS-0.6B` | ~1.2 GB | Voice cloning from audio samples | No |
| **CustomVoice** | `Qwen/Qwen3-TTS-0.6B-CustomVoice` | ~1.2 GB | Pre-trained bundled timbres | Yes |
| **VoiceDesign** | `Qwen/Qwen3-TTS-0.6B-VoiceDesign` | ~1.2 GB | Text-description voice generation | Yes |

### Tier Selection

Users can switch between tiers in TTS Settings. The tier can also be selected during installation for users with lower-spec machines.

**Important:** Embedding dimensions differ between tiers:
- 1.7B models: 2048-dimensional embeddings
- 0.6B models: 1024-dimensional embeddings

Embeddings created with one tier are **not compatible** with the other tier.

### Memory Management

- Only **one model loaded at a time** (lazy loading strategy)
- Quality tier models require ~3.4 GB RAM/VRAM each
- Small tier models require ~1.2 GB RAM/VRAM each
- Models are downloaded on first use from HuggingFace Hub
- Cached locally in `~/.cache/huggingface/` (default) or custom path

### Tier-Aware Embedding Fallback

When generating with an embedding voice and the current model tier doesn't match the embedding dimensions, MyVoice automatically falls back to real-time voice cloning:

1. Detects tensor dimension mismatch error
2. Locates `source_audio.wav` in the emotion folder
3. Falls back to `generate_voice_clone()` using that audio
4. Uses `ref_text` from embedding metadata if available, otherwise x-vector mode

This ensures voices work across tiers, though with slightly different quality characteristics.

### Model Selection Logic

```
Voice Type Selected → Model Loaded
─────────────────────────────────
BUNDLED voice      → CustomVoice model
DESIGNED voice     → VoiceDesign model
CLONED voice       → Base model
```

## Voice Types

### Bundled Voices (CustomVoice Model)

9 pre-trained speaker timbres with full emotion control:

| Speaker | Language | Gender | Style |
|---------|----------|--------|-------|
| Vivian | English | Female | Warm, friendly |
| Serena | English | Female | Clear, professional |
| Dylan | English | Male | Young, energetic |
| Eric | English | Male | Deep, mature |
| Ryan | English | Male | Casual, conversational |
| Aiden | English | Male | Neutral, calm |
| Uncle_Fu | Mandarin | Male | Traditional |
| Ono_Anna | Japanese | Female | Soft, feminine |
| Sohee | Korean | Female | Modern, clear |

**Usage in MyVoice:**
```python
# CustomVoice model automatically loaded
voice_profile = VoiceProfile(
    voice_type=VoiceType.BUNDLED,
    speaker_id="Vivian"
)
```

### Designed Voices (VoiceDesign Model)

Create custom voices from text descriptions. Supports emotion control.

**Description Guidelines:**
- Keep descriptions under 100 words
- Focus on: age, gender, tone, speaking style, accent
- Avoid contradictory attributes
- Be specific but not overly complex

**Example Descriptions:**
```
"A warm, elderly grandmother with a gentle British accent"
"Young professional woman, confident and articulate, American"
"Deep male voice, calm and reassuring, news anchor style"
```

**Usage in MyVoice:**
```python
voice_profile = VoiceProfile(
    voice_type=VoiceType.DESIGNED,
    voice_prompt="Warm elderly grandmother with gentle British accent"
)
```

### Cloned Voices (Base Model)

Clone voices from audio samples. **No emotion control available.**

**Audio Requirements:**
- Duration: 3-10 seconds (3 seconds minimum)
- Format: WAV, MP3, FLAC
- Quality: Clear speech, minimal background noise
- Content: Representative sample of target voice

**Cloning-Instruction Trade-off:**
The Base model faces a fundamental trade-off between voice cloning fidelity and instruction following (emotion control). High-fidelity cloning requires the model to strictly replicate the reference audio, which conflicts with modifying prosody for emotions.

**Usage in MyVoice:**
```python
voice_profile = VoiceProfile(
    voice_type=VoiceType.CLONED,
    reference_audio_path="path/to/sample.wav"
)
```

## Emotion Control

### Supported Presets

| Preset | Description | Available For |
|--------|-------------|---------------|
| Neutral | Default, natural tone | BUNDLED, DESIGNED |
| Happy | Upbeat, cheerful | BUNDLED, DESIGNED |
| Sad | Somber, melancholic | BUNDLED, DESIGNED |
| Angry | Intense, forceful | BUNDLED, DESIGNED |
| Flirtatious | Playful, teasing | BUNDLED, DESIGNED |

### Custom Emotion Prompts

Beyond presets, users can provide custom emotion descriptions:

```
"Speak with nervous excitement, slightly trembling"
"Confident and authoritative, like a CEO"
"Exhausted but trying to stay positive"
```

### Emotion Support Matrix

| Voice Type | Presets | Custom Prompts | Reason |
|------------|---------|----------------|--------|
| BUNDLED | Yes | Yes | CustomVoice model trained for instruction following |
| DESIGNED | Yes | Yes | VoiceDesign model supports emotional modification |
| CLONED | No | No | Base model prioritizes voice fidelity over instructions |

## Performance Specifications

### Latency

| Metric | Value | Notes |
|--------|-------|-------|
| First packet | ~97 ms | Streaming mode enabled |
| Full generation | ~2-5 s | Depends on text length |
| Model loading | ~10-30 s | First use, cold start |
| Model switching | ~10-30 s | When changing voice types |

### Streaming Architecture

MyVoice uses streaming TTS for responsive playback:

1. Text submitted to TTS engine
2. Audio chunks generated incrementally
3. First chunk plays within ~97ms
4. Remaining chunks stream as generated
5. Dual-stream output (monitor + virtual mic)

### Quality Settings

| Sample Rate | Channels | Bit Depth |
|-------------|----------|-----------|
| 24000 Hz | Mono | 16-bit |

Audio is upsampled to 48000 Hz for output compatibility.

## Integration Architecture

### Service Layer

```
QwenTTSService (Primary)
├── ModelLoadingManager (Lazy loading, model switching)
├── AudioNormalizer (Volume normalization)
└── StreamingBuffer (Chunk management)

AudioCoordinator (Output)
├── MonitorAudioService (Speaker output)
└── VirtualMicrophoneService (Virtual mic routing)
```

### File Structure

```
voice_files/
├── embeddings/           # Saved voice embeddings (v3.0 structure)
│   └── {voice_name}/
│       ├── metadata.json
│       ├── preview.wav
│       └── {emotion}/
│           ├── 1.7/
│           │   └── embedding.pt    # Quality tier embedding
│           ├── 0.6/
│           │   └── embedding.pt    # Small tier embedding
│           └── source_audio.wav    # Shared source for fallback
├── design_sessions/     # Voice Design working files
└── reference_audio/     # Original reference recordings

~/.cache/huggingface/    # Downloaded models (default)
├── Qwen--Qwen3-TTS-1.7B/
├── Qwen--Qwen3-TTS-1.7B-CustomVoice/
├── Qwen--Qwen3-TTS-1.7B-VoiceDesign/
├── Qwen--Qwen3-TTS-0.6B/
├── Qwen--Qwen3-TTS-0.6B-CustomVoice/
└── Qwen--Qwen3-TTS-0.6B-VoiceDesign/
```

## Error Handling

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `OutOfMemoryError` | Insufficient RAM | Close other applications, use 16GB+ RAM |
| `Model not found` | Network/download issue | Check internet, retry download |
| `Invalid reference audio` | Bad audio file | Use 3-10s clear speech sample |
| `Generation timeout` | Text too long | Split into shorter segments |

### Graceful Degradation

1. **Model load failure** → Retry with exponential backoff
2. **Generation failure** → Return error with user-friendly message
3. **Audio device failure** → Continue with available device

## Best Practices

### For Voice Cloning

1. Record in quiet environment
2. Use consistent microphone distance
3. Speak naturally (not reading)
4. Provide 5-7 second samples for best results
5. Test with short phrases first

### For Voice Design

1. Start with simple descriptions
2. Iterate based on output
3. Save successful designs as profiles
4. Avoid conflicting attributes

### For Performance

1. Preload commonly used voices
2. Keep text segments under 200 words
3. Use appropriate emotion presets (faster than custom)
4. Monitor memory usage during extended sessions

## Configuration

### Environment Variables

```bash
# Custom model cache location
HF_HOME=/path/to/models

# Disable GPU (force CPU)
CUDA_VISIBLE_DEVICES=""

# Enable debug logging
MYVOICE_DEBUG=1
```

### Application Settings

Settings stored in `%LOCALAPPDATA%\MyVoice\config.json`:

```json
{
  "tts": {
    "default_voice": "Vivian",
    "default_emotion": "neutral",
    "streaming_enabled": true,
    "chunk_size": 1024,
    "model_tier": "quality"
  }
}
```

| Setting | Values | Description |
|---------|--------|-------------|
| `model_tier` | `"quality"`, `"small"` | Model tier selection (1.7B or 0.6B) |

## References

- [Qwen3-TTS Official Repository](https://github.com/QwenLM/Qwen3-TTS)
- [HuggingFace Model Cards](https://huggingface.co/Qwen)
- MyVoice Architecture: `_bmad-output/planning-artifacts/architecture.md`
- Service Implementation: `src/myvoice/services/qwen_tts_service.py`

---

*Last Updated: 2026-02-20*
*MyVoice V2.5.0*
