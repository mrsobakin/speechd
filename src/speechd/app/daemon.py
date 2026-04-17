import fcntl
import logging
import os
import signal
import subprocess
from importlib.metadata import version as pkg_version
from pathlib import Path

from pydantic import BaseModel, model_validator, Field

from speechd.pipeline import Pipeline
from speechd.app.recorder import AudioRecorder

logger = logging.getLogger(__name__)


class Daemon:
    class Config(BaseModel):
        typer: tuple[str, ...]
        recorder: AudioRecorder.Config = Field(default_factory=AudioRecorder.Config)

        runtime_dir: Path = Field(default_factory=lambda: Path(os.environ.get("XDG_RUNTIME_DIR", "/tmp")))
        indicator_file: Path = None
        pidfile: Path = None

        @model_validator(mode="after")
        def set_derived_paths(self) -> "Config":
            if self.indicator_file is None:
                self.indicator_file = self.runtime_dir / "speechd.recording"
            if self.pidfile is None:
                self.pidfile = self.runtime_dir / "speechd.pid"
            return self

    def __init__(self, pipeline: Pipeline, config: Config | None = None):
        self._config = config or self.Config()
        self._pipeline = pipeline

        self._recorder = AudioRecorder(self._config.recorder)
        self._pidfile_fd: int | None = None

    def _acquire_pidfile(self) -> bool:
        fd = os.open(self._config.pidfile, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError):
            os.close(fd)
            return False
        os.truncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
        self._pidfile_fd = fd
        return True

    def cleanup(self):
        if self._pidfile_fd is not None:
            try:
                fcntl.flock(self._pidfile_fd, fcntl.LOCK_UN)
                os.close(self._pidfile_fd)
                self._pidfile_fd = None
            except Exception:
                pass
        try:
            self._config.indicator_file.unlink(missing_ok=True)
        except Exception:
            pass

    def _type_text(self, text: str):
        if not text:
            return

        try:
            subprocess.run(list(self._config.typer), input=text.encode(), check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to type text: {e}")
        except FileNotFoundError:
            logger.error(f"Typer not found: {self._config.typer[0]}")

    def _wait_result(self) -> str:
        self._recorder.wait()
        recording = self._recorder.get_result()

        if recording.timed_out:
            logger.info(f"Recording cancelled: exceeded {self._config.recorder.timeout}s timeout")
            return ""
        if not recording.has_audio:
            return ""

        return self._pipeline.transcribe(recording.audio)

    def run(self):
        if not self._acquire_pidfile():
            logger.error("Another instance is already running")
            raise SystemExit(1)

        signal.signal(signal.SIGCONT, lambda *_: self._recorder.stop())
        signal.signal(signal.SIGTERM, lambda *_: (self.cleanup(), exit(0)))
        signal.signal(signal.SIGINT, lambda *_: (self.cleanup(), exit(0)))

        logger.info(f"speechd v{pkg_version('speechd')} Ready. PID: {os.getpid()}")
        logger.info(
            f"Model: {self._pipeline.config.engine.model}, Language: {self._pipeline.config.engine.language or 'auto-detect'}"
        )
        logger.info(f"Typer: {' '.join(self._config.typer)}")
        logger.info(f"Timeout: {self._config.recorder.timeout}s")

        while True:
            os.kill(os.getpid(), signal.SIGSTOP)
            self._config.indicator_file.touch()
            self._recorder.start()
            self._type_text(self._wait_result())
            self._config.indicator_file.unlink(missing_ok=True)
