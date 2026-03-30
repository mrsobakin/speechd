from typing import Any

import numpy as np


_PROCESSOR_REGISTRY: dict[str, type] = {}


def register(name: str):
    def decorator(cls):
        _PROCESSOR_REGISTRY[name] = cls
        return cls

    return decorator


class PrePipeline:
    def __init__(self, *processors):
        self.processors = processors

    @classmethod
    def from_configs(cls, configs: list[Any]) -> "PrePipeline":
        processors = []
        for cfg in configs:
            cfg_dict = dict(cfg)
            processor_type = cfg_dict.pop("type")
            processor_cls = _PROCESSOR_REGISTRY[processor_type]
            config = processor_cls.Config.model_validate(cfg_dict)
            processors.append(processor_cls(config=config))
        return cls(*processors)

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio_float = audio.astype(np.float32) / 32768.0

        for processor in self.processors:
            audio_float = processor.process(audio_float)
            if len(audio_float) == 0:
                break

        return (audio_float * 32768.0).astype(np.int16)
