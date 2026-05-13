# TTS Invalid Token Range Error Troubleshooting

## Problem
```
[WARNING] audio_utils: Parse: Token 4 outside valid range [10, 28682). Rejecting.
[WARNING] audio_utils: Parse: Filtered 1 raw codes down to 0 valid codes.
```

**Tokens received: 1, 2, 3, 4, 5, 6** (invalid - should be 10-28682)

---

## Root Causes

### 1. Wrong TTS Model
The server is not running the Orpheus model. Valid models:
- ✅ `orpheus-3b-0.1-ft` (expected)
- ❌ Other models (llama, mistral, etc.)

**Check:**
```bash
# Verify the TTS server is using Orpheus model
python debug_tts_response.py
```

**Fix:**
```python
# In config.py or environment variable
TTS_MODEL = "orpheus-3b-0.1-ft"
```

---

### 2. Wrong Server URL
The server at `http://127.0.0.1:1234` is not the Orpheus server.

**Check:**
```python
# In config.py
SERVER_BASE_URL = os.getenv("SERVER_BASE_URL", "http://127.0.0.1:1234")
```

**Verify server is running:**
```bash
# Windows PowerShell
Test-NetConnection localhost -Port 1234

# Linux/Mac
netstat -an | grep 1234
```

**Fix - Set correct server:**
```bash
# If using different port/host
set SERVER_BASE_URL=http://your-server:port
python main.py
```

---

### 3. Response Format Mismatch
The server response format doesn't match expected format.

**Expected format:**
```json
{
  "choices": [{
    "delta": {
      "content": "<custom_token_15> <custom_token_22> ..."
    }
  }]
}
```

**What we're getting:**
```json
{
  "content": "1 2 3 4 5 6"  // or similar
}
```

**Debug:**
```bash
python debug_tts_response.py | head -50
```

---

### 4. Server Not Returning GGUF Tokens
The response might be returning plain text instead of `<custom_token_N>` format.

**Expected:**
```
<custom_token_10> <custom_token_15> <custom_token_22>
```

**Getting:**
```
hello world
```

**This means:**
- Server is not in TTS mode / not generating audio tokens
- Server is in text completion mode (not audio)
- Wrong endpoint configuration

---

## Step-by-Step Debugging

### Step 1: Test Server Connectivity
```bash
python debug_tts_response.py
```

Look for:
- ✓ "Streaming Response:" → Server is responding
- ✗ "Connection Error" → Server not running
- ✗ "Timeout Error" → Server is slow or not responding

### Step 2: Check Response Format
Run debug script and check "Parsed JSON" section:
- Does it have `content` or `choices`/`delta`/`content`?
- What's in the content field?
- Are there `<custom_token_N>` patterns?

### Step 3: Verify Token Format
If you see tokens like:
- ✓ `<custom_token_15>` → Correct format ✓
- ❌ `1 2 3` → Wrong format ✗
- ❌ Plain text → Wrong mode ✗

### Step 4: Check Model & Prompt
The prompt must be correct for Orpheus:
```python
# config.py
TTS_PROMPT_FORMAT = "<|audio|>{voice}: {text}<|eot_id|>"
```

This tells the model to generate audio tokens, not text.

---

## Solutions by Issue

### Issue: Tokens 1-6 (Single Digits)
**Likely Cause:** Server returning metadata or error codes, not audio tokens

**Solution:**
```python
# Debug what's being returned
python debug_tts_response.py

# If you see plain numbers, the server isn't generating audio
# The Orpheus model should return <custom_token_N> where N >= 10
```

---

### Issue: Plain Text Response
**Example:** Server returns "the quick brown fox"

**Likely Cause:** 
- Server is in text-completion mode, not audio mode
- Wrong model loaded
- Prompt format not recognized

**Solution:**
```python
# Verify config.py has:
TTS_PROMPT_FORMAT = "<|audio|>{voice}: {text}<|eot_id|>"
TTS_MODEL = "orpheus-3b-0.1-ft"
```

---

### Issue: Empty/No Response
**Likely Cause:** Server not running or endpoint wrong

**Solution:**
```bash
# 1. Check if server running
netstat -an | grep 1234  # Linux/Mac
Get-NetTCPConnection -LocalPort 1234  # Windows

# 2. Test endpoint directly
curl http://127.0.0.1:1234/v1/completions

# 3. Check firewall
# Port 1234 should be open locally
```

---

### Issue: Valid Tokens Received But No Audio
If tokens look valid (e.g., `<custom_token_15>`) but no audio:

**Possible causes:**
- SNAC model not loaded for decoding
- Invalid SNAC token mapping
- Insufficient tokens received

**Check logs for:**
```
SNAC model loaded successfully → ✓ SNAC is ready
redistribute_codes: Decode successful → ✓ Tokens were decoded
Failed to decode → ✗ Problem with token decoding
```

---

## Quick Checklist

- [ ] TTS server running at http://127.0.0.1:1234?
- [ ] Server is using `orpheus-3b-0.1-ft` model?
- [ ] Correct TTS_MODEL in config.py?
- [ ] Correct TTS_PROMPT_FORMAT in config.py?
- [ ] Response contains `<custom_token_N>` patterns?
- [ ] Tokens in valid range [10, 28682)?
- [ ] SNAC model loaded for decoding?

---

## Testing the Full Pipeline

```bash
# 1. Start the FastAPI server
python main.py

# 2. In another terminal, test TTS endpoint
curl -X POST http://localhost:8000/api/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "hello world",
    "voice": "tara"
  }' \
  --output test_audio.wav

# 3. Play the audio
# If you get a valid audio file, TTS is working!
# If empty or error, check the logs
```

---

## Environment Variables

Set these if using non-default values:

```bash
# External TTS server
set SERVER_BASE_URL=http://localhost:1234

# TTS model name
set TTS_MODEL=orpheus-3b-0.1-ft

# Example with custom values
set SERVER_BASE_URL=http://192.168.1.100:8000
set TTS_MODEL=orpheus-3b-0.1-ft
python main.py
```

---

## Need More Help?

1. **Run debug script:**
   ```bash
   python debug_tts_response.py > tts_debug.log 2>&1
   ```

2. **Share the output showing:**
   - TTS Server URL
   - First few lines of response
   - Any error messages

3. **Check if the external server:**
   - Is actually an Orpheus server
   - Has the right model loaded
   - Is accessible from localhost:1234
