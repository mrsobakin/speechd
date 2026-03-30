from speechd.speechd import Speechd
import fcntl
import logging
import os
import signal
import subprocess
from pathlib import Path

from speechd.config import Config

logger = logging.getLogger(__name__)


class SpeechDaemon:
    def __init__(self, config: Config):
        self.config = config
        self.speechd = Speechd(config)

        self.runtime_dir = Path(config.runtime_dir)
        self.indicator_file = self.runtime_dir / "speechd.recording"
        self.pidfile = self.runtime_dir / "speechd.pid"
        self._pidfile_fd: int | None = None

    def _acquire_pidfile(self) -> bool:
        fd = os.open(self.pidfile, os.O_RDWR | os.O_CREAT, 0o644)
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
            self.indicator_file.unlink(missing_ok=True)
        except Exception:
            pass

    def _type_text(self, text: str):
        if not text:
            return

        try:
            subprocess.run(list(self.config.typer), input=text.encode(), check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to type text: {e}")
        except FileNotFoundError:
            logger.error(f"Typer not found: {self.config.typer[0]}")

    def run(self):
        if not self._acquire_pidfile():
            logger.error("Another instance is already running")
            raise SystemExit(1)

        signal.signal(signal.SIGCONT, lambda *_: self.speechd.stop())
        signal.signal(signal.SIGTERM, lambda *_: (self.cleanup(), exit(0)))
        signal.signal(signal.SIGINT, lambda *_: (self.cleanup(), exit(0)))

        logger.info(f"Ready. PID: {os.getpid()}")
        logger.info(f"Model: {self.config.model}, Language: {self.config.language or 'auto-detect'}")
        logger.info(f"Typer: {' '.join(self.config.typer)}")
        logger.info(f"Timeout: {self.config.timeout_seconds}s")

        while True:
            os.kill(os.getpid(), signal.SIGSTOP)
            self.indicator_file.touch()
            self.speechd.start()
            self._type_text(self.speechd.wait_result())
            self.indicator_file.unlink(missing_ok=True)
