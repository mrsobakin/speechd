import numpy as np


class AGC:
    def __init__(self, target_rms: float = 0.2, max_gain_db: float = 30.0):
        self.target_rms = target_rms
        self.max_gain_db = max_gain_db

    def process(self, audio: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(audio**2))
        if rms < 1e-9:
            return audio

        gain = self.target_rms / rms
        max_gain = 10 ** (self.max_gain_db / 20)
        gain = min(gain, max_gain)

        return np.clip(audio * gain, -1.0, 1.0)
