import atexit
import fcntl
import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import sounddevice as sd

from speechd.audio import VoiceActivityDetector
from speechd.config import Config
from speechd.transcribe import Transcriber

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
        )
        self.recording = False
        self.frames: list[np.ndarray] = []
        self.stream: sd.InputStream | None = None
        self.recording_start_time: float | None = None
        self.timeout_cancelled = False

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

    def toggle(self):
        if not self.recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        logger.info("Recording started")
        self.frames = []
        self.recording = True
        self.timeout_cancelled = False
        self.recording_start_time = time.monotonic()
        self.indicator_file.touch()
        self.stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=1,
            dtype=np.int16,
            callback=self._audio_callback,
        )
        self.stream.start()

    def _stop_recording(self, timeout: bool = False):
        self.recording = False
        self.recording_start_time = None
        try:
            self.indicator_file.unlink(missing_ok=True)
        except Exception:
            pass

        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

        if timeout:
            logger.info(f"Recording cancelled: exceeded {self.config.timeout_seconds}s timeout")
            return

        if not self.frames:
            return

        audio_data = np.concatenate(self.frames)
        duration = len(audio_data) / self.config.sample_rate
        logger.info(f"Processing {duration:.1f}s of audio...")

        t0 = time.monotonic()
        audio_clean = self.vad.process(audio_data)
        vad_time = time.monotonic() - t0

        if len(audio_clean) == 0:
            logger.info(f"No speech detected (VAD: {vad_time:.2f}s)")
            return

        logger.info(f"Transcribing... (VAD: {vad_time:.2f}s)")
        t1 = time.monotonic()
        result = self.transcriber.transcribe(audio_clean)
        transcribe_time = time.monotonic() - t1

        if result.success and result.text:
            logger.info(f"[{transcribe_time:.2f}s] {result.text}")
            self._type_text(result.text)

    def _audio_callback(self, indata, _frames, _time, _status):
        if not self.recording:
            return

        if self.recording_start_time is not None:
            elapsed = time.monotonic() - self.recording_start_time
            if elapsed > self.config.timeout_seconds:
                self.timeout_cancelled = True
                self._stop_recording(timeout=True)
                return

        self.frames.append(indata.copy().flatten())

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
                subprocess.run(["wtype", "-"], input=text.encode(), check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to type text: {e}")
            except FileNotFoundError:
                logger.error("wtype not found - cannot type text")

    def run(self):
        if not self._acquire_pidfile():
            logger.error("Another instance is already running")
            raise SystemExit(1)

        signal.signal(signal.SIGUSR1, lambda *_: self.toggle())
        signal.signal(signal.SIGTERM, lambda *_: (self.cleanup(), exit(0)))
        signal.signal(signal.SIGINT, lambda *_: (self.cleanup(), exit(0)))
        atexit.register(self.cleanup)

        logger.info(f"Ready. PID: {os.getpid()}")
        logger.info(f"Model: {self.config.model}, Language: {self.config.language}")
        logger.info(f"Timeout: {self.config.timeout_seconds}s")

        while True:
            signal.pause()
