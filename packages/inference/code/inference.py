"""
Custom inference handler for Cohere Transcribe on SageMaker.

Receives audio as base64-encoded JSON and returns transcription text.
"""

import base64
import json
import logging
import os
import tempfile

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

logger = logging.getLogger(__name__)

LANGUAGE = os.environ.get("TRANSCRIBE_LANGUAGE", "ja")


def model_fn(model_dir: str) -> dict:
    """Load model and processor from HuggingFace Hub."""
    model_id = os.environ.get("HF_MODEL_ID", "CohereLabs/cohere-transcribe-03-2026")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    logger.info("Loading processor from %s", model_id)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    logger.info("Loading model from %s to %s", model_id, device)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, trust_remote_code=True
    ).to(device)
    model.eval()

    logger.info("Model loaded successfully on %s", device)
    return {"model": model, "processor": processor}


def input_fn(request_body: bytes, request_content_type: str) -> dict:
    """Deserialize input data.

    Supported content types:
    - application/json: {"audio_base64": "<base64>", "language": "ja"}
    - audio/wav, audio/flac, audio/mp3, audio/ogg: raw binary audio
    """
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        audio_bytes = base64.b64decode(payload["audio_base64"])
        language = payload.get("language", LANGUAGE)
        return {"audio_bytes": audio_bytes, "language": language}

    if request_content_type.startswith("audio/"):
        return {"audio_bytes": request_body, "language": LANGUAGE}

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: dict, model_dict: dict) -> str:
    """Run transcription on the audio input."""
    model = model_dict["model"]
    processor = model_dict["processor"]

    audio_bytes = input_data["audio_bytes"]
    language = input_data["language"]

    suffix = ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(audio_bytes)
        tmp_path = f.name

    try:
        texts = model.transcribe(
            processor=processor,
            audio_files=[tmp_path],
            language=language,
        )
        return texts[0] if texts else ""
    finally:
        os.unlink(tmp_path)


def output_fn(prediction: str, accept: str) -> str:
    """Serialize prediction output."""
    if accept == "application/json":
        return json.dumps({"transcription": prediction}, ensure_ascii=False)
    return prediction
