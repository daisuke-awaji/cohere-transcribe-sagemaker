"""
Test client for the WebSocket streaming transcription server.

Usage:
  python test_client.py ws://<EC2_IP>:8765 audio.wav --language en
"""

import argparse
import asyncio
import json
import wave

import websockets


async def stream_audio(uri: str, audio_path: str, language: str, chunk_seconds: float):
    async with websockets.connect(uri) as ws:
        # Configure
        await ws.send(json.dumps({
            "command": "configure",
            "language": language,
            "format": "pcm",
        }))
        config_resp = await ws.recv()
        print(f"Config: {config_resp}")

        # Read WAV file
        with wave.open(audio_path, "rb") as wf:
            sample_rate = wf.getframerate()
            sample_width = wf.getsampwidth()
            channels = wf.getnchannels()
            n_frames = wf.getnframes()
            duration = n_frames / sample_rate

            print(f"Audio: {duration:.1f}s, {sample_rate}Hz, {sample_width * 8}bit, {channels}ch")

            chunk_frames = int(sample_rate * chunk_seconds)
            sent_frames = 0

            while True:
                data = wf.readframes(chunk_frames)
                if not data:
                    break
                sent_frames += chunk_frames
                elapsed = min(sent_frames / sample_rate, duration)
                print(f"  Sending chunk: {elapsed:.1f}s / {duration:.1f}s ({len(data)} bytes)")
                await ws.send(data)

                # Check for any available responses (non-blocking)
                try:
                    resp = await asyncio.wait_for(ws.recv(), timeout=0.1)
                    result = json.loads(resp)
                    print(f"  >> {result.get('transcription', '')}")
                except asyncio.TimeoutError:
                    pass

        # Signal end
        await ws.send(json.dumps({"command": "end"}))

        # Collect remaining responses
        while True:
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=30)
                result = json.loads(resp)
                text = result.get("transcription", "")
                if text:
                    print(f"  >> {text}")
                if result.get("is_final"):
                    print("Done.")
                    break
            except asyncio.TimeoutError:
                print("Timeout waiting for response.")
                break


def main():
    parser = argparse.ArgumentParser(description="Test streaming transcription")
    parser.add_argument("uri", help="WebSocket URI (e.g., ws://1.2.3.4:8765)")
    parser.add_argument("audio", help="Path to WAV audio file")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--chunk-seconds", type=float, default=3.0, help="Chunk duration in seconds")
    args = parser.parse_args()

    asyncio.run(stream_audio(args.uri, args.audio, args.language, args.chunk_seconds))


if __name__ == "__main__":
    main()
