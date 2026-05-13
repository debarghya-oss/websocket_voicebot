#!/usr/bin/env python3
"""
Test script to stream TTS audio via WebSocket and play it locally.
"""
import asyncio
import json
import base64
import numpy as np
from websockets.client import connect
import sounddevice as sd


async def test_tts_websocket():
    """Test TTS WebSocket streaming and play audio."""
    
    uri = "ws://localhost:8000/ws/tts"
    
    # TTS request payload
    tts_request = {
        "text": "Hello! This is a test of the websocket audio streaming. Can you hear me?",
        "voice": "tara",
        "tts_temperature": 2.0,
        "tts_top_p": 0.9,
        "tts_repetition_penalty": 1.1,
        "buffer_groups": 5,
        "padding_ms": 0,
        "min_decode_batch_groups": 7
    }
    
    print(f"Connecting to {uri}")
    print(f"Sending TTS request: {tts_request['text']}")
    
    audio_chunks = []
    sample_rate = 24000
    
    async with connect(uri) as websocket:
        # Send TTS request
        await websocket.send(json.dumps(tts_request))
        print("✓ Sent TTS request")
        
        # Receive audio chunks
        while True:
            message_str = await websocket.recv()
            message = json.loads(message_str)
            
            msg_type = message.get("type")
            print(f"[{msg_type}]", end=" ", flush=True)
            
            if msg_type == "tts_started":
                sample_rate = message.get("sample_rate", 24000)
                encoding = message.get("encoding", "unknown")
                print(f"Sample Rate: {sample_rate}Hz, Encoding: {encoding}")
                
            elif msg_type == "tts_audio_chunk":
                # Decode base64 audio
                audio_base64 = message.get("audio", "")
                if audio_base64:
                    audio_bytes = base64.b64decode(audio_base64)
                    float32_array = np.frombuffer(audio_bytes, dtype=np.float32)
                    audio_chunks.append(float32_array)
                    print(f"Chunk size: {len(float32_array)} samples", flush=True)
                    
            elif msg_type == "tts_done":
                print("✓ Streaming complete")
                break
                
            elif msg_type == "error":
                print(f"Error: {message.get('message')}")
                break
    
    # Combine all audio chunks
    if audio_chunks:
        full_audio = np.concatenate(audio_chunks)
        print(f"\n✓ Received {len(audio_chunks)} chunks, total samples: {len(full_audio)}")
        print(f"Duration: {len(full_audio) / sample_rate:.2f} seconds")
        
        # Play audio
        print(f"\nPlaying audio...")
        try:
            sd.play(full_audio, sample_rate)
            sd.wait()  # Wait for playback to complete
            print("✓ Playback complete")
        except Exception as e:
            print(f"Playback error: {e}")
            print("   Install sounddevice: pip install sounddevice")
    else:
        print("No audio chunks received")


async def test_stt_websocket(audio_file_path: str = None):
    """Test STT WebSocket with audio upload."""
    
    if not audio_file_path:
        print("For STT testing, provide an audio file path")
        return
    
    uri = "ws://localhost:8000/ws/stt"
    
    # Read audio file
    try:
        with open(audio_file_path, "rb") as f:
            audio_bytes = f.read()
    except FileNotFoundError:
        print(f"Audio file not found: {audio_file_path}")
        return
    
    # Encode to base64
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    stt_request = {
        "engine": "indic",
        "language": "bn",
        "decode_mode": "ctc",
        "audio": audio_base64
    }
    
    print(f"Connecting to {uri}")
    print(f"Sending STT request with {len(audio_bytes)} bytes of audio")
    
    async with connect(uri) as websocket:
        await websocket.send(json.dumps(stt_request))
        print("✓ Sent STT request")
        
        while True:
            message_str = await websocket.recv()
            message = json.loads(message_str)
            
            msg_type = message.get("type")
            print(f"[{msg_type}]", end=" ", flush=True)
            
            if msg_type == "stt_processing":
                print(message.get("message"))
                
            elif msg_type == "stt_result":
                text = message.get("text")
                engine = message.get("engine")
                language = message.get("language")
                print(f"\n✓ Transcription: {text}")
                print(f"  Engine: {engine}, Language: {language}")
                break
                
            elif msg_type == "error":
                print(f"Error: {message.get('message')}")
                break


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "stt":
        # Test STT with audio file
        audio_file = sys.argv[2] if len(sys.argv) > 2 else None
        asyncio.run(test_stt_websocket(audio_file))
    else:
        # Test TTS (default)
        asyncio.run(test_tts_websocket())
