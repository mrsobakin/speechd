import numpy as np


class PrePipeline:
    def __init__(self, *processors):
        self.processors = processors

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio_float = audio.astype(np.float32) / 32768.0

        for processor in self.processors:
            audio_float = processor.process(audio_float)
            if len(audio_float) == 0:
                break

        return (audio_float * 32768.0).astype(np.int16)
