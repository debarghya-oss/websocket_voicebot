# STT Model Loading Troubleshooting Guide

## Error: "STT service unavailable: Whisper model not loaded"

This means the Whisper model failed to load during server startup. Here's how to fix it:

## Quick Diagnostics

**Step 1: Run the diagnostic script**
```bash
python diagnose_stt.py
```

This will check:
- ✓ Whisper library installation
- ✓ PyTorch/CUDA availability
- ✓ Model loading capability
- ✓ Configuration settings

---

## Common Issues & Solutions

### Issue 1: Whisper Not Installed

**Error in logs:**
```
✗ Whisper library not found
```

**Solution:**
```bash
pip install -U openai-whisper
```

---

### Issue 2: OpenAI Whisper vs Other Packages

Make sure you install the correct package:

```bash
# CORRECT - OpenAI's Whisper
pip install -U openai-whisper

# INCORRECT - different package
pip install whisper  # ❌ Don't use this one
```

**Verify correct installation:**
```python
python -c "import whisper; print(whisper.__file__)"
```

---

### Issue 3: PyTorch/CUDA Issues

**Error in logs:**
```
CUDA out of memory | CUDA is not available
```

**Solution A: Force CPU mode**
```bash
# Edit config.py
# Change: DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# To:     DEVICE = "cpu"
```

**Solution B: Reinstall PyTorch for your setup**
```bash
# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

### Issue 4: Model Download Fails

**Error in logs:**
```
Error: Connection timeout | 404 Not Found
```

**Causes:**
- Network issues
- Hugging Face is down
- Disk space full

**Solutions:**
```bash
# Check disk space
df -h  # Linux/Mac
dir   # Windows

# Pre-download model manually
python -c "import whisper; whisper.load_model('base.en')"

# Use smaller model to test
# Edit config.py: WHISPER_MODEL_NAME = "tiny.en"
```

---

### Issue 5: Model Not Initialized in Memory

**Symptoms:**
- Diagnostic passes but model still not loaded at runtime
- Works after restart
- Intermittent failures

**Solution: Check initialization order**

In `main_refactored.py`, verify the model is loaded before starting the server:

```python
if __name__ == "__main__":
    logger.info("--- Loading AI Models ---")
    
    # Load models BEFORE starting server
    stt_engine.whisper_model = stt_engine.load_whisper_model()
    if stt_engine.whisper_model is None:
        logger.critical("FAILED TO LOAD WHISPER MODEL!")
        sys.exit(1)  # Exit if model fails to load
    
    # Start server
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### Issue 6: Wrong Model Name

**Error in logs:**
```
Error: Unknown model: base-en | model 'base-en' not found
```

**Valid model names:**
- `tiny.en` - English only, smallest
- `base.en` - English only, fast
- `small.en` - English only, better quality
- `medium.en` - English only, high quality
- `base` - Multi-language (includes Bengali)
- `small` - Multi-language
- `medium` - Multi-language
- `large` - Multi-language, largest

**For Bengali transcription, use multi-language models:**
```python
# In config.py
WHISPER_MODEL_NAME = "base"  # ✓ Supports Bengali
# NOT
WHISPER_MODEL_NAME = "base.en"  # ✗ English only
```

---

## Complete Diagnostics

### 1. Check logs for initialization errors
```bash
# During startup, watch for:
# ✓ "Whisper library imported successfully"
# ✓ "Loading Whisper model..."
# ✓ "Whisper model (...) loaded successfully"
```

### 2. Test Whisper directly
```python
import whisper
model = whisper.load_model("base")
result = model.transcribe("audio.mp3", language="bn")
print(result["text"])
```

### 3. Run diagnostic
```bash
python diagnose_stt.py
```

### 4. Check configuration
```python
# Verify config.py
from config import WHISPER_MODEL_NAME, DEVICE, TEMP_AUDIO_DIR
print(f"Model: {WHISPER_MODEL_NAME}")
print(f"Device: {DEVICE}")
print(f"Temp Dir: {TEMP_AUDIO_DIR}")
```

### 5. Monitor startup logs
```bash
python main_refactored.py 2>&1 | tee startup.log
# Check startup.log for detailed errors
```

---

## If Still Not Working

**Collect debug info:**
```bash
# 1. Run diagnostic
python diagnose_stt.py > diagnostic_output.txt 2>&1

# 2. Check Python version
python --version

# 3. List installed packages
pip list | grep -i whisper
pip list | grep -i torch
pip list | grep -i openai

# 4. Save startup logs
python main_refactored.py > startup.log 2>&1

# Share these files for debugging
```

---

## Quick Fix Checklist

- [ ] `pip install -U openai-whisper` (not just `whisper`)
- [ ] Use **multi-language model** for Bengali: `base` or `small`
- [ ] Check device: CPU or CUDA?
- [ ] Disk space available?
- [ ] Network connectivity?
- [ ] Run `python diagnose_stt.py`
- [ ] Check server startup logs
- [ ] Restart server after fixing

---

## Model Loading Performance

**Typical loading times:**

| Model | Size | Time (CPU) | Time (GPU) | Memory |
|-------|------|-----------|-----------|--------|
| tiny | 40MB | ~5s | ~1s | 1GB |
| base | 140MB | ~15s | ~2s | 2GB |
| small | 480MB | ~30s | ~5s | 5GB |
| medium | 1.5GB | ~60s | ~10s | 10GB |

**First run:** Model downloads from Hugging Face (~1-2 minutes)  
**Subsequent runs:** Model loads from cache (~5-15 seconds)

---

## Bengali Language Notes

For Bengali transcription, you need:

✓ **Multi-language model**: `base`, `small`, `medium`, `large`  
✗ **English-only model**: `base.en`, `small.en`, `medium.en`

**Current config (stt_engine.py):**
```python
result = whisper_model.transcribe(tmp_audio_file_path, language="bn", ...)
```

This forces Bengali output, but the **base model** supports it better.
