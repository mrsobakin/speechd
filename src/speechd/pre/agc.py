import numpy as np
import pyloudnorm as pyln
from pydantic import BaseModel

from speechd.pre.pipeline import register


@register("agc")
class AGC:
    class Config(BaseModel):
        target_loudness: float = -23.0

    def __init__(self, config: Config | None = None):
        config = config or self.Config()
        self.meter = pyln.Meter(16000)
        self.target_loudness = config.target_loudness

    def process(self, audio: np.ndarray) -> np.ndarray:
        if len(audio) == 0:
            return audio

        loudness = self.meter.integrated_loudness(audio)
        if loudness == -float("inf"):
            return audio

        audio = pyln.normalize.loudness(audio, loudness, self.target_loudness)
        return np.clip(audio, -1.0, 1.0).astype(np.float32)
