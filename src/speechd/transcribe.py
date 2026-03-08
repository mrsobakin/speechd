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
    def __init__(self, api_key: str, model: str, language: str | None, audio_quality: float):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.language = language
        self.audio_quality = audio_quality

    def transcribe(self, audio_data: np.ndarray) -> TranscriptionResult:
        if len(audio_data) == 0:
            return TranscriptionResult(text="", success=True)

        try:
            opus_data = self._encode_opus(audio_data)
            buffer = io.BytesIO(opus_data)
            buffer.name = "audio.ogg"

            logger.debug(f"Transcribing {len(audio_data) / 16000:.1f}s of audio")

            if self.language:
                result = self.client.audio.transcriptions.create(
                    file=buffer,
                    model=self.model,
                    language=self.language,
                    response_format="text",
                )
            else:
                result = self.client.audio.transcriptions.create(
                    file=buffer,
                    model=self.model,
                    response_format="text",
                )

            return TranscriptionResult(text=str(result), success=True)
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return TranscriptionResult(text="", success=False, error=str(e))

    def _normalize_rms(self, audio: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
        rms = np.sqrt(np.mean(audio**2))
        if rms > 1e-8:
            audio = audio * (target_rms / rms)
            np.clip(audio, -1.0, 1.0, out=audio)
        return audio

    def _encode_opus(self, audio_data: np.ndarray) -> bytes:
        audio_float = audio_data.astype(np.float32) / 32768.0
        audio_float = self._normalize_rms(audio_float)
        buf = io.BytesIO()
        sf.write(
            buf,
            audio_float,
            16000,
            format="OGG",
            subtype="OPUS",
            compression_level=self.audio_quality,
        )
        return buf.getvalue()
