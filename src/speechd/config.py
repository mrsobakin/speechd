import logging
import os
import stat
from pathlib import Path
from typing import Any

import tomllib
from pydantic import BaseModel, Field, ValidationError, field_validator


def get_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "speechd" / "config.toml"
    return Path.home() / ".config" / "speechd" / "config.toml"


class Config(BaseModel, frozen=True):
    api_key: str
    typer: tuple[str, ...]
    model: str = "whisper-large-v3-turbo"
    language: str | None = None
    prompt: str | None = None
    timeout_seconds: int = Field(default=300, alias="timeout")
    audio_quality: float = 0.8
    runtime_dir: str = Field(default_factory=lambda: os.environ.get("XDG_RUNTIME_DIR", "/tmp"))
    pre: list[dict[str, Any]] = []
    post: list[dict[str, Any]] = []

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or v == "your-api-key-here":
            raise ValueError("must be set")
        return v

    @classmethod
    def load(cls) -> "Config":
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

        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise RuntimeError(cls._format_error(e)) from None

    @staticmethod
    def _format_error(e: ValidationError) -> str:
        lines = ["Config validation failed:"]
        for err in e.errors():
            field = ".".join(str(loc) for loc in err["loc"])
            msg = err["msg"]
            if err["type"] == "missing":
                lines.append(f"  - {field}: required field missing")
            else:
                input_val = err.get("input", "")
                if isinstance(input_val, str) and len(input_val) > 20:
                    input_val = input_val[:20] + "..."
                lines.append(f"  - {field}: {msg} (got: {repr(input_val)})")
        return "\n".join(lines)
