from speechd.pipeline import Pipeline
from speechd.app.daemon import Daemon
import logging
import os
import stat
from pathlib import Path

import tomllib
from pydantic import BaseModel, ValidationError


def get_config_path() -> Path:
    config_home = os.environ.get("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home) / "speechd" / "config.toml"
    return Path.home() / ".config" / "speechd" / "config.toml"


class Config(BaseModel, frozen=True):
    daemon: Daemon.Config
    pipeline: Pipeline.Config

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
