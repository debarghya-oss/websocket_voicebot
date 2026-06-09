# Modular FastAPI Project Structure

This project has been refactored into modular components for better maintainability, testability, and reusability.

## Module Organization

### Core Modules

#### `config.py`
- **Purpose**: Centralized configuration and constants
- **Contains**: 
  - Environment variables
  - API endpoints
  - Audio processing constants (SNAC, Orpheus codec parameters)
  - Device setup
  - Model names and paths

#### `models.py`
- **Purpose**: Pydantic models for request/response validation
- **Contains**:
  - `TTSRequest` - Text-to-Speech request model
  - `STTResponse` - Speech-to-Text response model

#### `audio_utils.py`
- **Purpose**: Audio processing utility functions
- **Contains**:
  - `parse_gguf_codes()` - Parse audio tokens from TTS API response
  - `redistribute_codes()` - Convert codes to SNAC format and decode audio
  - `apply_fade()` - Apply fade in/out to audio chunks

#### `tts_engine.py`
- **Purpose**: Text-to-Speech functionality
- **Contains**:
  - `load_snac_model()` - Load SNAC model for TTS
  - `generate_speech_stream_bytes()` - Stream audio generation
  - Global `snac_model` instance

#### `stt_engine.py`
- **Purpose**: Speech-to-Text functionality
- **Contains**:
  - `load_whisper_model()` - Load Whisper model for STT
  - `transcribe_audio()` - Transcribe audio file to Bengali text
  - Global `whisper_model` instance

#### `main.py` (or `main_refactored.py`)
- **Purpose**: FastAPI application and endpoints
- **Contains**:
  - FastAPI app configuration
  - Route definitions for TTS/STT
  - Model loading and startup logic
  - Uvicorn server startup

### Legacy Module
- `llm_router.py` - LLM routing functionality (unchanged)

## Usage

### Starting the Server
```bash
python main_refactored.py
```

### File Structure
```
Orpheus_FastAPI/
├── config.py              # Configuration constants
├── models.py              # Pydantic models
├── audio_utils.py         # Audio processing utilities
├── tts_engine.py          # TTS engine
├── stt_engine.py          # STT engine (Bengali)
├── main_refactored.py     # FastAPI app (new)
├── main.py                # Original (keep for backup)
├── llm_router.py          # LLM routing
├── requirements.txt
├── static/
│   ├── index.html
│   └── script.js
└── temp_stt_audio_files/
```

## Benefits of Modular Architecture

1. **Separation of Concerns** - Each module has a single responsibility
2. **Testability** - Modules can be tested independently
3. **Reusability** - Engines can be used in other projects
4. **Maintainability** - Easier to locate and fix bugs
5. **Scalability** - Easy to add new features or engines

## API Endpoints

### TTS (Text-to-Speech)
```
POST /api/tts/stream
Content-Type: application/json

{
  "text": "Hello world",
  "voice": "tara",
  "tts_temperature": 0.9,
  "tts_top_p": 0.9,
  "tts_repetition_penalty": 1.1,
  "buffer_groups": 1,
  "padding_ms": 0,
  "min_decode_batch_groups": 10
}

Response: Audio stream (audio/octet-stream)
Headers: X-Sample-Rate: 24000, X-Audio-Format: FLOAT32_PCM
```

### STT (Speech-to-Text) - Bengali
```
POST /api/stt/transcribe
Content-Type: multipart/form-data

audio_file: <binary audio file>

Response: 
{
  "text": "transcribed bengali text",
  "language": "bn",
  "error": null
}
```

## Environment Variables

```bash
# Server Configuration
SERVER_BASE_URL=http://127.0.0.1:1234

# TTS Model
TTS_MODEL=orpheus-3b-0.1-ft

# STT Model
WHISPER_MODEL_NAME=base.en

# Device (auto-detected, set to "cpu" to force CPU)
# DEVICE is automatically set based on CUDA availability
```

## Future Improvements

- [ ] Add caching for frequently transcribed audio
- [ ] Support multiple languages (beyond Bengali)
- [ ] Add batch processing for TTS
- [ ] Implement request queuing and priority
- [ ] Add metrics and monitoring
- [ ] Implement API authentication
