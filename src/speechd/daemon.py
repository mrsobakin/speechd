import fcntl
import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from speechd.audio import VoiceActivityDetector
from speechd.config import Config
from speechd.transcribe import Transcriber
from speechd.recorder import AudioRecorder, RecordingResult

logger = logging.getLogger(__name__)


class SpeechDaemon:
    def __init__(self, config: Config):
        self.config = config
        logger.info("Loading VAD model...")
        self.vad = VoiceActivityDetector(sample_rate=config.sample_rate)
        self.transcriber = Transcriber(
            api_key=config.groq_api_key,
            model=config.model,
            language=config.language,
            sample_rate=config.sample_rate,
            audio_quality=config.audio_quality,
        )
        self.recorder = AudioRecorder(
            sample_rate=config.sample_rate,
            timeout_seconds=config.timeout_seconds,
        )

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

    def _process(self, result: RecordingResult):
        if result.timed_out:
            logger.info(f"Recording cancelled: exceeded {self.config.timeout_seconds}s timeout")
            return
        if not result.has_audio:
            return

        duration = len(result.audio) / self.config.sample_rate
        logger.info(f"Processing {duration:.1f}s of audio...")

        t0 = time.monotonic()
        audio_clean = self.vad.process(result.audio)
        vad_time = time.monotonic() - t0

        if len(audio_clean) == 0:
            logger.info(f"No speech detected (VAD: {vad_time:.2f}s)")
            return

        logger.info(f"Transcribing... (VAD: {vad_time:.2f}s)")
        t1 = time.monotonic()
        transcription = self.transcriber.transcribe(audio_clean)
        transcribe_time = time.monotonic() - t1

        if transcription.success and transcription.text:
            logger.info(f"[{transcribe_time:.2f}s] {transcription.text}")
            self._type_text(transcription.text)

    @staticmethod
    def _postprocess_text(text: str) -> str:
        text = text.replace("—", "-")
        text = text.replace("–", "-")
        text = re.sub(r"(\s)-(\s)", r"\1--\2", text)
        return text

    def _type_text(self, text: str):
        text = self._postprocess_text(text).strip().replace("\n", " ")
        if text:
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

        signal.signal(signal.SIGCONT, lambda *_: self.recorder.stop())
        signal.signal(signal.SIGTERM, lambda *_: (self.cleanup(), exit(0)))
        signal.signal(signal.SIGINT, lambda *_: (self.cleanup(), exit(0)))

        logger.info(f"Ready. PID: {os.getpid()}")
        logger.info(
            f"Model: {self.config.model}, Language: {self.config.language or 'auto-detect'}"
        )
        logger.info(f"Typer: {' '.join(self.config.typer)}")
        logger.info(f"Timeout: {self.config.timeout_seconds}s")

        while True:
            os.kill(os.getpid(), signal.SIGSTOP)
            self.indicator_file.touch()
            self.recorder.start()
            self.recorder.wait()
            self.indicator_file.unlink(missing_ok=True)
            self._process(self.recorder.get_result())
