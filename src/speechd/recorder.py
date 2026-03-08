import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


@dataclass
class RecordingResult:
    audio: np.ndarray | None
    timed_out: bool = False

    @property
    def has_audio(self) -> bool:
        return self.audio is not None and len(self.audio) > 0


class AudioRecorder:
    def __init__(self, timeout_seconds: float):
        self.timeout_seconds = timeout_seconds

        self._frames: list[np.ndarray] = []
        self._stream: sd.InputStream | None = None
        self._start_time: float | None = None
        self._done = threading.Event()
        self._timed_out = False

    def start(self):
        self._frames = []
        self._timed_out = False
        self._done.clear()
        self._start_time = time.monotonic()
        self._stream = sd.InputStream(
            samplerate=16000,
            channels=1,
            dtype=np.int16,
            callback=self._callback,
        )
        self._stream.start()
        logger.info("Recording started")

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._start_time = None
        self._done.set()

    def wait(self):
        self._done.wait()

    def get_result(self) -> RecordingResult:
        if self._timed_out or not self._frames:
            return RecordingResult(audio=None, timed_out=self._timed_out)
        return RecordingResult(audio=np.concatenate(self._frames))

    def _callback(self, indata, _frames, _time, _status):
        if self._start_time is not None:
            if time.monotonic() - self._start_time > self.timeout_seconds:
                self._timed_out = True
                self.stop()
                return
        self._frames.append(indata.copy().flatten())
