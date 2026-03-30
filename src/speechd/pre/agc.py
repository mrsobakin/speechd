import numpy as np
import pyloudnorm as pyln


class AGC:
    def __init__(self, target_loudness: float = -23.0):
        self.meter = pyln.Meter(16000)
        self.target_loudness = target_loudness

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        loudness = self.meter.integrated_loudness(audio)
        if loudness == -float("inf"):
            return audio

        audio = pyln.normalize.loudness(audio, loudness, self.target_loudness)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)
