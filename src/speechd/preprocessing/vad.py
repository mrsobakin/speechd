import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    def __init__(self, sample_rate: int = 16000, max_silence_ms: int = 500):
        self.sample_rate = sample_rate
        self.max_silence_samples = int(max_silence_ms * sample_rate / 1000)
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

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        audio_tensor = torch.from_numpy(audio)

        with torch.no_grad():
            get_speech_ts = self.utils[0]
            speech_timestamps = get_speech_ts(
                audio_tensor,
                self.model,
                sampling_rate=self.sample_rate,
                threshold=0.5,
                min_speech_duration_ms=150,
                min_silence_duration_ms=100,
            )

        if not speech_timestamps:
            return np.array([], dtype=np.float32)

        segments = []
        prev_end = 0

        for ts in speech_timestamps:
            silence_len = ts["start"] - prev_end
            if silence_len > 0:
                silence_samples = min(silence_len, self.max_silence_samples)
                segments.append(np.zeros(silence_samples, dtype=np.float32))

            segments.append(audio[ts["start"] : ts["end"]])
            prev_end = ts["end"]

        return np.concatenate(segments) if segments else np.array([], dtype=np.float32)
