import logging
import os
import stat
from dataclasses import dataclass
from pathlib import Path


def get_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "speechd" / "config.toml"
    return Path.home() / ".config" / "speechd" / "config.toml"


@dataclass(frozen=True)
class Config:
    groq_api_key: str
    model: str
    language: str
    sample_rate: int
    timeout_seconds: int
    runtime_dir: str
    audio_quality: float

    @classmethod
    def load(cls) -> "Config":
        import tomllib

        config_path = get_config_path()

        if not config_path.exists():
            raise RuntimeError(f"Config not found at {config_path}")

        mode = config_path.stat().st_mode
        if mode & (stat.S_IRGRP | stat.S_IROTH):
            logging.warning(
                "WARNING: %s is world accessible. Consider limiting its permissions (600).",
                config_path,
            )

        with open(config_path, "rb") as f:
            data = tomllib.load(f)

        api_key = data.get("api_key", "")
        if not api_key or api_key == "your-api-key-here":
            raise RuntimeError(f"Please set api_key in {config_path}")

        return cls(
            groq_api_key=api_key,
            model=data.get("model", "whisper-large-v3-turbo"),
            language=data.get("language", "ru"),
            sample_rate=data.get("sample_rate", 16000),
            timeout_seconds=data.get("timeout", 300),
            runtime_dir=os.environ.get("XDG_RUNTIME_DIR", "/tmp"),
            audio_quality=data.get("audio_quality", 0.8),
        )
