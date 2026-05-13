# Migration Guide: From Monolithic to Modular Architecture

## Overview
The original `main.py` (603 lines) has been split into 5 focused modules + 1 refactored main file.

## Module Breakdown

| Module | Lines | Purpose |
|--------|-------|---------|
| `config.py` | ~65 | Configuration & constants |
| `models.py` | ~23 | Pydantic request/response models |
| `audio_utils.py` | ~164 | Audio processing functions |
| `tts_engine.py` | ~230 | TTS model & streaming |
| `stt_engine.py` | ~60 | STT model & transcription |
| `main_refactored.py` | ~120 | FastAPI app & endpoints |
| **Total** | **~662** | **+59 lines (mostly docstrings/spacing)** |

## Key Changes

### 1. Configuration Centralization
**Before**: Constants scattered throughout main.py
**After**: All constants in `config.py`, imported as needed

```python
# Before
DEFAULT_TTS_VOICE = ALL_VOICES[0]
ORPHEUS_MIN_ID = 10

# After
from config import DEFAULT_TTS_VOICE, ORPHEUS_MIN_ID
```

### 2. Model Loading
**Before**: Model loading in main.py startup
**After**: Dedicated functions in engine modules

```python
# Before
if SNAC is not None:
    snac_model = SNAC.from_pretrained(...)

# After (main_refactored.py)
tts_engine.snac_model = tts_engine.load_snac_model()
```

### 3. Request/Response Models
**Before**: In main.py as TTSRequest, STTResponse
**After**: Separate `models.py`

### 4. Audio Processing
**Before**: Helper functions in main.py
**After**: Separate `audio_utils.py` for reusability

## How to Switch

### Option 1: Replace main.py (if you're confident)
```bash
cd Orpheus_FastAPI
mv main.py main_backup.py
mv main_refactored.py main.py
python main.py
```

### Option 2: Run side-by-side (safer)
```bash
python main_refactored.py
# Original still available if needed
```

### Option 3: Gradual Migration
Keep using `main.py`, but start importing from new modules:
```python
# In your main.py
from config import DEFAULT_TTS_VOICE
from models import TTSRequest, STTResponse
```

## Benefits

✅ **Easier Testing**: Test each module independently  
✅ **Better Reusability**: Use TTS/STT engines in other projects  
✅ **Cleaner Code**: Each file has a single responsibility  
✅ **Easier Debugging**: Find bugs in focused files  
✅ **Scalability**: Add new features without bloating main.py  

## Troubleshooting

### Import Errors
```
ImportError: No module named 'config'
```
**Solution**: Make sure you're running from `Orpheus_FastAPI/` directory

### Model Not Loading
```
SNAC model not loaded at startup. TTS will be unavailable.
```
**Solution**: Check logs - likely missing dependencies:
```bash
pip install git+https://github.com/hubertsiuzdak/snac.git
pip install -U openai-whisper
```

### Module Circular Imports
If you get circular import errors, check that modules only import from `config.py` at the top level.

## File Dependencies

```
config.py
  ↓
models.py ← (imports Optional from typing)
audio_utils.py ← (imports from config)
tts_engine.py ← (imports from config, models, audio_utils)
stt_engine.py ← (imports from config, models)
main_refactored.py ← (imports from all above)
```

## Performance
No performance change - same code, better organization.

## Next Steps (Optional)

1. **Add Unit Tests**:
   ```bash
   pytest test_audio_utils.py
   pytest test_tts_engine.py
   ```

2. **Add Logging Module**:
   ```python
   # logging_config.py
   def setup_logging():
       ...
   ```

3. **Add Environment Validation**:
   ```python
   # validators.py
   def validate_env_vars():
       ...
   ```

4. **Create API Client**:
   ```python
   # client.py
   class OrpheusClient:
       def __init__(self, base_url="http://localhost:8000"):
           ...
   ```
