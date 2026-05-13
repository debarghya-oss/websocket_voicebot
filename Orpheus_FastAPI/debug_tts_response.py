import requests
import json
from config import TTS_API_ENDPOINT, TTS_MODEL, TTS_PROMPT_FORMAT, TTS_PROMPT_STOP_TOKENS, STREAM_HEADERS, STREAM_TIMEOUT_SECONDS

print("=" * 80)
print("TTS API RESPONSE DEBUGGING")
print("=" * 80)

print(f"\nTTS Server: {TTS_API_ENDPOINT}")
print(f"TTS Model: {TTS_MODEL}")

# Test payload
test_text = "hello world"
voice = "tara"

payload = {
    "model": TTS_MODEL,
    "prompt": TTS_PROMPT_FORMAT.format(voice=voice, text=test_text),
    "temperature": 0.9,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "n_predict": -1,
    "stop": TTS_PROMPT_STOP_TOKENS,
    "stream": True
}

print(f"\nPayload:")
print(json.dumps(payload, indent=2))

print("\n" + "=" * 80)
print("Streaming Response:")
print("=" * 80)

try:
    response = requests.post(
        TTS_API_ENDPOINT, 
        json=payload, 
        headers=STREAM_HEADERS, 
        stream=True, 
        timeout=STREAM_TIMEOUT_SECONDS
    )
    response.raise_for_status()
    
    line_count = 0
    for line in response.iter_lines():
        if not line:
            continue
        
        line_count += 1
        try:
            decoded_line = line.decode('utf-8', errors='ignore')
            print(f"\n[Line {line_count}]")
            print(f"Raw: {decoded_line[:200]}")
            
            if decoded_line.startswith("data:"):
                json_str = decoded_line[5:].strip()
                if json_str:
                    try:
                        data = json.loads(json_str)
                        print(f"Parsed JSON: {json.dumps(data, indent=2)[:500]}")
                        
                        # Check content
                        if "content" in data:
                            content = data["content"]
                            print(f"Content field: {content[:200]}")
                        elif "choices" in data and data["choices"]:
                            choice = data["choices"][0]
                            if "delta" in choice:
                                delta = choice.get("delta", {})
                                content = delta.get("content", "") or choice.get("text", "")
                                print(f"Delta/text content: {content[:200]}")
                    except json.JSONDecodeError as e:
                        print(f"JSON parse error: {e}")
        except Exception as e:
            print(f"Error: {e}")
        
        if line_count > 10:  # Limit output
            print("\n... (showing first 10 lines)")
            break
    
    print(f"\nTotal lines received: {line_count}")
    
except requests.exceptions.ConnectionError as e:
    print(f" Connection Error: {e}")
    print("   Check if the TTS server is running at http://127.0.0.1:1234")
except requests.exceptions.Timeout as e:
    print(f" Timeout Error: {e}")
except requests.exceptions.RequestException as e:
    print(f" Request Error: {e}")
except Exception as e:
    print(f" Unexpected Error: {e}")
    import traceback
    traceback.print_exc()
