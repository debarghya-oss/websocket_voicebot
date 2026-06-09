import sys
import os

print("=" * 80)
print("WHISPER STT DIAGNOSTIC CHECK")
print("=" * 80)

# Step 1: Check Whisper import
print("\n[1] Checking Whisper library import...")
try:
    import whisper
    print("    ✓ Whisper imported successfully")
    print(f"    Whisper version: {whisper.__version__ if hasattr(whisper, '__version__') else 'Unknown'}")
    print(f"    Whisper location: {whisper.__file__}")
except ImportError as e:
    print(f"    ✗ FAILED to import Whisper: {e}")
    print("    Install with: pip install -U openai-whisper")
    sys.exit(1)

# Step 2: Check PyTorch/CUDA
print("\n[2] Checking PyTorch and device availability...")
try:
    import torch
    print(f"    ✓ PyTorch imported successfully (version {torch.__version__})")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"    Device: {device}")
    if device == "cuda":
        print(f"    GPU: {torch.cuda.get_device_name(0)}")
        print(f"    CUDA version: {torch.version.cuda}")
except ImportError:
    print("    ✗ PyTorch not found")
    sys.exit(1)

# Step 3: Check available models
print("\n[3] Checking available Whisper model sizes...")
try:
    from whisper import _MODELS
    available_models = list(_MODELS.keys()) if hasattr(whisper, '_MODELS') else ["Unknown"]
    print(f"    Available models: {available_models}")
except:
    print("    ✓ Model info not directly accessible (this is OK)")

# Step 4: Try to load a model
print("\n[4] Attempting to load Whisper model (base.en)...")
try:
    print("    Loading model (this may take a minute on first run)...")
    test_model = whisper.load_model("base.en", device=device)
    print(f"    ✓ Model loaded successfully!")
    print(f"    Model type: {type(test_model)}")
    print(f"    Model device: {next(test_model.parameters()).device}")
except Exception as e:
    print(f"    ✗ FAILED to load model: {type(e).__name__}: {e}")
    print(f"    Traceback:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Check config
print("\n[5] Checking config.py settings...")
try:
    from config import WHISPER_MODEL_NAME, DEVICE, TEMP_AUDIO_DIR
    print(f"    WHISPER_MODEL_NAME: {WHISPER_MODEL_NAME}")
    print(f"    DEVICE: {DEVICE}")
    print(f"    TEMP_AUDIO_DIR: {TEMP_AUDIO_DIR}")
    if os.path.exists(TEMP_AUDIO_DIR):
        print(f"    ✓ Temp directory exists")
    else:
        print(f"    ⚠ Temp directory does not exist (will be created at runtime)")
except Exception as e:
    print(f"    ✗ Error loading config: {e}")

# Step 6: Check STT engine
print("\n[6] Checking stt_engine.py...")
try:
    from whisper_stt_engine import load_whisper_model
    print(f"    ✓ stt_engine imported successfully")
    print(f"    Attempting to load model through stt_engine...")
    loaded = load_whisper_model()
    if loaded is not None:
        print(f"    ✓ Model loaded successfully through stt_engine")
    else:
        print(f"    ✗ load_whisper_model() returned None")
        sys.exit(1)
except Exception as e:
    print(f"    ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✓ ALL DIAGNOSTICS PASSED - Whisper STT is ready!")
print("=" * 80)
print("\nYou can now run: python main_refactored.py")
