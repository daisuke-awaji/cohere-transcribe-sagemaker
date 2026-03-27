"""
WebSocket relay server for real-time audio transcription.

Receives audio chunks via WebSocket, buffers them,
and sends to local vLLM server for transcription.

Protocol:
  Client -> Server:
    - Binary frames: raw audio data (WAV/PCM chunks)
    - JSON text: {"command": "transcribe", "language": "ja"}
    - JSON text: {"command": "end"} to flush buffer and get final result

  Server -> Client:
    - JSON text: {"transcription": "...", "is_final": true/false}
"""

import asyncio
import io
import json
import logging
import os
import tempfile
import time
import wave

import aiohttp
import websockets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:8000")
BUFFER_SECONDS = float(os.environ.get("BUFFER_SECONDS", "5"))
SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", "16000"))
SAMPLE_WIDTH = 2  # 16-bit PCM
CHANNELS = 1
WS_PORT = int(os.environ.get("WS_PORT", "8765"))


def pcm_to_wav(pcm_data: bytes, sample_rate: int = SAMPLE_RATE) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


async def transcribe_audio(audio_bytes: bytes, language: str) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        async with aiohttp.ClientSession() as session:
            data = aiohttp.FormData()
            data.add_field(
                "file",
                open(tmp_path, "rb"),
                filename="audio.wav",
                content_type="audio/wav",
            )
            data.add_field("model", "CohereLabs/cohere-transcribe-03-2026")
            data.add_field("language", language)

            url = f"{VLLM_URL}/v1/audio/transcriptions"
            async with session.post(url, data=data, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("text", "")
                else:
                    error = await resp.text()
                    logger.error("vLLM error %d: %s", resp.status, error)
                    return f"[ERROR: {resp.status}]"
    finally:
        os.unlink(tmp_path)


async def handle_client(websocket):
    client_id = id(websocket)
    logger.info("Client %s connected", client_id)

    audio_buffer = bytearray()
    language = "ja"
    last_flush_time = time.time()
    input_format = "pcm"  # "pcm" or "wav"

    async def flush_buffer():
        nonlocal audio_buffer, last_flush_time
        if not audio_buffer:
            return
        chunk = bytes(audio_buffer)
        audio_buffer.clear()
        last_flush_time = time.time()

        if input_format == "pcm":
            wav_data = pcm_to_wav(chunk)
        else:
            wav_data = chunk

        logger.info("Transcribing %d bytes of audio (lang=%s)", len(wav_data), language)
        text = await transcribe_audio(wav_data, language)
        if text:
            await websocket.send(json.dumps({
                "transcription": text,
                "is_final": False,
            }))

    try:
        async for message in websocket:
            if isinstance(message, str):
                data = json.loads(message)
                cmd = data.get("command", "")

                if cmd == "configure":
                    language = data.get("language", language)
                    input_format = data.get("format", input_format)
                    logger.info("Client %s configured: lang=%s, format=%s", client_id, language, input_format)
                    await websocket.send(json.dumps({"status": "configured"}))

                elif cmd == "end":
                    await flush_buffer()
                    await websocket.send(json.dumps({
                        "transcription": "",
                        "is_final": True,
                    }))

                elif cmd == "transcribe":
                    await flush_buffer()

            elif isinstance(message, bytes):
                audio_buffer.extend(message)
                buffer_duration = len(audio_buffer) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)
                if buffer_duration >= BUFFER_SECONDS:
                    await flush_buffer()

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client %s disconnected", client_id)
    except Exception as e:
        logger.exception("Error handling client %s: %s", client_id, e)


async def main():
    logger.info("Starting WebSocket server on port %d", WS_PORT)
    logger.info("vLLM URL: %s", VLLM_URL)
    logger.info("Buffer duration: %s seconds", BUFFER_SECONDS)

    async with websockets.serve(handle_client, "0.0.0.0", WS_PORT):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    asyncio.run(main())
