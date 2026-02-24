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
    def __init__(self, api_key: str, model: str, language: str, sample_rate: int):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.language = language
        self.sample_rate = sample_rate

    def transcribe(self, audio_data: np.ndarray) -> TranscriptionResult:
        if len(audio_data) == 0:
            return TranscriptionResult(text="", success=True)

        try:
            opus_data = self._encode_opus(audio_data)
            buffer = io.BytesIO(opus_data)
            buffer.name = "audio.ogg"

            logger.debug(f"Transcribing {len(audio_data) / self.sample_rate:.1f}s of audio")
            result = self.client.audio.transcriptions.create(
                file=buffer,
                model=self.model,
                language=self.language,
                response_format="text",
            )
            return TranscriptionResult(text=result, success=True)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(text="", success=False, error=str(e))

    def _encode_opus(self, audio_data: np.ndarray) -> bytes:
        audio_float = audio_data.astype(np.float32) / 32768.0
        buf = io.BytesIO()
        sf.write(
            buf,
            audio_float,
            self.sample_rate,
            format="OGG",
            subtype="OPUS",
            compression_level=0.8,
        )
        return buf.getvalue()
