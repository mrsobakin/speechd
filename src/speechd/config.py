import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = """api_key = "your-grok-api-key"
model = "whisper-large-v3-turbo"
language = "ru"
sample_rate = 16000
timeout = 300
"""


@dataclass(frozen=True)
class Config:
    groq_api_key: str
    model: str
    language: str
    sample_rate: int
    timeout_seconds: int
    runtime_dir: str

    @classmethod
    def load(cls) -> "Config":
        import tomllib

        config_path = Path.home() / ".config" / "speechd" / "config.toml"

        if not config_path.exists():
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(DEFAULT_CONFIG)
            config_path.chmod(0o600)
            raise RuntimeError(
                f"Config created at {config_path}\nPlease edit and add your Groq API key"
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
        )
