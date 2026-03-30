from typing import Any


_PROCESSOR_REGISTRY: dict[str, type] = {}


def register(name: str):
    def decorator(cls):
        _PROCESSOR_REGISTRY[name] = cls
        return cls

    return decorator


class PostPipeline:
    def __init__(self, *processors):
        self.processors = processors

    @classmethod
    def from_configs(cls, configs: list[Any]) -> "PostPipeline":
        processors = []
        for cfg in configs:
            cfg_dict = dict(cfg)
            processor_type = cfg_dict.pop("type")
            processor_cls = _PROCESSOR_REGISTRY[processor_type]
            config = processor_cls.Config.model_validate(cfg_dict)
            processors.append(processor_cls(config=config))
        return cls(*processors)

    def process(self, text: str) -> str:
        for processor in self.processors:
            text = processor.process(text)

        return text
