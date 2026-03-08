import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        logger.info("Loading Silero VAD model...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
            skip_validation=True,
        )
        self.model.eval()
        logger.info("VAD model loaded")

    def process(self, audio_data: np.ndarray) -> np.ndarray:
        if len(audio_data) == 0:
            return audio_data

        audio_float = audio_data.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float)

        with torch.no_grad():
            get_speech_ts = self.utils[0]
            speech_timestamps = get_speech_ts(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
            )

        if not speech_timestamps:
            return np.array([], dtype=np.int16)

        result = np.zeros_like(audio_data)
        for ts in speech_timestamps:
            result[ts["start"] : ts["end"]] = audio_data[ts["start"] : ts["end"]]
        return result
