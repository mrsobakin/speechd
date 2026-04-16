import logging
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
from pydantic import BaseModel

from speechd.engine import GroqEngine
from speechd.post import PostChain
from speechd.pre import PreChain

logger = logging.getLogger(__name__)


@contextmanager
def _measure_time(who: str):
    start = time.monotonic()
    yield
    took = time.monotonic() - start
    logger.info(f"{who} done in {took:.2f}s")


class Pipeline:
    class Config(BaseModel):
        pre: list[Any]
        post: list[Any]
        engine: GroqEngine.Config

    def __init__(self, config: Config):
        self.config = config
        self.pre = PreChain.from_configs(config.pre)
        self.post = PostChain.from_configs(config.post)
        self.engine = GroqEngine(config.engine)

    def transcribe(self, audio: np.ndarray) -> str:
        if len(audio) == 0:
            return ""

        duration = len(audio) / 16000
        logger.info(f"Processing {duration:.1f}s of audio...")

        with _measure_time("Preprocessing"):
            audio = self.pre.process(audio)

        if len(audio) == 0:
            logger.info(f"No speech detected")
            return ""

        with _measure_time("Transcription"):
            transcription = self.engine.transcribe(audio)

        if not transcription.success or not transcription.text:
            return ""

        logger.info(f'Raw transcription: "{transcription.text}"')

        with _measure_time("Postprocessing"):
            text = self.post.process(transcription.text)

        logger.info(f'Final transcription: "{text}"')

        return text.strip().replace("\n", " ")
