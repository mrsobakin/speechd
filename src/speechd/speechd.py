import logging
import numpy as np
import time
from contextlib import contextmanager

from speechd.pre import PrePipeline
from speechd.post import PostPipeline
from speechd.config import Config
from speechd.transcribe import Transcriber
from speechd.recorder import AudioRecorder

logger = logging.getLogger(__name__)


@contextmanager
def measure_time(who: str):
    start = time.monotonic()
    yield
    took = time.monotonic() - start
    logger.info(f"{who} done in {took:.2f}s")


class Speechd:
    def __init__(self, config: Config):
        self.config = config
        self.pre = PrePipeline.from_configs(config.pre)
        self.post = PostPipeline.from_configs(config.post)
        self.transcriber = Transcriber(
            api_key=config.api_key,
            model=config.model,
            language=config.language,
            prompt=config.prompt,
            audio_quality=config.audio_quality,
        )
        self.recorder = AudioRecorder(timeout_seconds=config.timeout_seconds)

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) == 0:
            return ""

        duration = len(audio) / 16000
        logger.info(f"Processing {duration:.1f}s of audio...")

        with measure_time("Preprocessing"):
            audio = self.pre.process(audio)

        if len(audio) == 0:
            logger.info(f"No speech detected")
            return ""

        with measure_time("Transcription"):
            transcription = self.transcriber.transcribe(audio)

        if not transcription.success or not transcription.text:
            return ""

        logger.info(f'Raw transcription: "{transcription.text}"')

        with measure_time("Postprocessing"):
            text = self.post.process(transcription.text)

        logger.info(f'Final transcription: "{text}"')

        return text.strip().replace("\n", " ")

    def start(self):
        self.recorder.start()

    def stop(self):
        self.recorder.stop()

    def wait_result(self) -> str:
        self.recorder.wait()
        recording = self.recorder.get_result()

        if recording.timed_out:
            logger.info(f"Recording cancelled: exceeded {self.config.timeout_seconds}s timeout")
            return ""
        if not recording.has_audio:
            return ""

        return self.transcribe(recording.audio)
