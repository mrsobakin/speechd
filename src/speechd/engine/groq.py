import io
import logging
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from groq import Groq
from pydantic import BaseModel

logger = logging.getLogger(__name__)


@dataclass
class TranscriptionResult:
    text: str
    success: bool
    error: str | None = None


class GroqEngine:
    class Config(BaseModel):
        api_key: str
        model: str = "whisper-large-v3-turbo"
        language: str | None = None
        prompt: str | None = None
        audio_quality: float = 0.8

    def __init__(self, config: Config):
        self.client = Groq(api_key=config.api_key)
        self.config = config

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        if len(audio) == 0:
            return TranscriptionResult(text="", success=True)

        try:
            opus_data = self._encode_opus(audio)
            buffer = io.BytesIO(opus_data)
            buffer.name = "audio.ogg"

            logger.debug(f"Transcribing {len(audio) / 16000:.1f}s of audio")

            kwargs = {
                "file": buffer,
                "model": self.config.model,
                "response_format": "text",
            }
            if self.config.language:
                kwargs["language"] = self.config.language
            if self.config.prompt:
                kwargs["prompt"] = self.config.prompt

            result = self.client.audio.transcriptions.create(**kwargs)

            return TranscriptionResult(text=str(result).strip(), success=True)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(text="", success=False, error=str(e))

    def _encode_opus(self, audio: np.ndarray) -> bytes:
        buf = io.BytesIO()
        sf.write(
            buf,
            audio,
            16000,
            format="OGG",
            subtype="OPUS",
            compression_level=self.config.audio_quality,
        )
        return buf.getvalue()
