"""
Standalone diagnostic: test_indic.py
-------------------------------------
Verifies the full Indic Conformer pipeline end-to-end:
  1. ffmpeg availability
  2. transformers / torch / torchaudio imports
  3. Model load from ai4bharat/indic-conformer-600m-multilingual
  4. Transcription of a real audio file (pass path as CLI arg)

Usage:
    python test_indic.py                           # model load only
    python test_indic.py path/to/audio.webm bn     # full transcription test
"""
import sys
import os
import subprocess
import tempfile
import time

SEP = "-" * 60

def check_ffmpeg():
    print(SEP)
    print("1. Checking ffmpeg ...")
    try:
        r = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        first_line = r.stdout.decode(errors="replace").split("\n")[0]
        print(f"   ✓ {first_line}")
        return True
    except FileNotFoundError:
        print("   ✗ ffmpeg NOT FOUND on PATH.")
        print("     Install: winget install Gyan.FFmpeg")
        print("     Then restart your terminal / VS Code.")
        return False

def check_imports():
    print(SEP)
    print("2. Checking Python dependencies ...")
    ok = True
    for pkg, mod in [
        ("transformers", "transformers"),
        ("torch",        "torch"),
        ("torchaudio",   "torchaudio"),
    ]:
        try:
            __import__(mod)
            print(f"   ✓ {pkg}")
        except ImportError as e:
            print(f"   ✗ {pkg} — {e}")
            ok = False
    return ok

def load_model():
    print(SEP)
    print("3. Loading Indic Conformer model ...")
    print("   (first run will download ~2 GB — be patient)")
    from transformers import AutoModel
    t0 = time.time()
    try:
        model = AutoModel.from_pretrained(
            "ai4bharat/indic-conformer-600m-multilingual",
            trust_remote_code=True,
        )
        print(f"   ✓ Model loaded in {time.time()-t0:.1f}s  |  type={type(model).__name__}")
        return model
    except Exception as e:
        print(f"   ✗ Model load failed: {type(e).__name__}: {e}")
        return None

def transcribe(model, audio_path: str, language: str = "bn"):
    import torch, torchaudio

    print(SEP)
    print(f"4. Transcribing: {audio_path!r}  (lang={language}) ...")

    ext = os.path.splitext(audio_path)[1].lower()
    needs_conv = ext not in {".wav", ".flac", ".mp3", ".ogg"}

    wav_path = audio_path
    if needs_conv:
        wav_path = audio_path + "_converted.wav"
        print(f"   → Converting {ext!r} to WAV via ffmpeg …")
        cmd = [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", "16000", "-ac", "1", "-f", "wav", wav_path,
        ]
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if r.returncode != 0:
            print(f"   ✗ ffmpeg error: {r.stderr.decode(errors='replace')[:300]}")
            return
        print(f"   ✓ Converted → {wav_path}")

    try:
        wav, sr = torchaudio.load(wav_path)
        wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        print(f"   ✓ Audio loaded  shape={tuple(wav.shape)}  sr_orig={sr}")

        t0 = time.time()
        result = model(wav, language, "ctc")
        print(f"   ✓ Transcription ({time.time()-t0:.2f}s): {result!r}")
    finally:
        if needs_conv and os.path.exists(wav_path):
            os.remove(wav_path)

def main():
    audio_file = sys.argv[1] if len(sys.argv) > 1 else None
    language   = sys.argv[2] if len(sys.argv) > 2 else "bn"

    print("=" * 60)
    print("  Indic Conformer — Diagnostic Script")
    print("=" * 60)

    ffmpeg_ok = check_ffmpeg()
    deps_ok   = check_imports()

    if not deps_ok:
        print("\n✗ Fix missing dependencies before continuing.")
        sys.exit(1)

    model = load_model()
    if model is None:
        print("\n✗ Model failed to load. Check your internet connection or HuggingFace cache.")
        sys.exit(1)

    if audio_file:
        if not os.path.exists(audio_file):
            print(f"\n✗ Audio file not found: {audio_file!r}")
            sys.exit(1)
        if not ffmpeg_ok and os.path.splitext(audio_file)[1].lower() not in {".wav", ".flac"}:
            print("\n✗ ffmpeg required to decode this format. Install it first.")
            sys.exit(1)
        transcribe(model, audio_file, language)
    else:
        print(SEP)
        print("4. (Skipped — no audio file provided)")
        print("   To test transcription: python test_indic.py your_audio.webm bn")

    print(SEP)
    print("✓ All checks passed." if (ffmpeg_ok and deps_ok and model) else
          "⚠ Some checks failed — see above.")

if __name__ == "__main__":
    main()
