import io
import logging
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from groq import Groq

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    success: bool
    error: str | None = None


class Transcriber:
    def __init__(self, api_key: str, model: str, language: str | None, prompt: str | None, audio_quality: float):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.language = language
        self.prompt = prompt
        self.audio_quality = audio_quality

    def transcribe(self, audio_data: np.ndarray) -> TranscriptionResult:
        if len(audio_data) == 0:
            return TranscriptionResult(text="", success=True)

        try:
            opus_data = self._encode_opus(audio_data)
            buffer = io.BytesIO(opus_data)
            buffer.name = "audio.ogg"

            logger.debug(f"Transcribing {len(audio_data) / 16000:.1f}s of audio")

            kwargs = {
                "file": buffer,
                "model": self.model,
                "response_format": "text",
            }
            if self.language:
                kwargs["language"] = self.language
            if self.prompt:
                kwargs["prompt"] = self.prompt

            result = self.client.audio.transcriptions.create(**kwargs)

            return TranscriptionResult(text=str(result).strip(), success=True)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(text="", success=False, error=str(e))

    def _encode_opus(self, audio_data: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(
            buf,
            audio_data,
            16000,
            format="OGG",
            subtype="OPUS",
            compression_level=self.audio_quality,
        )
        return buf.getvalue()
