import logging
import signal

import numpy as np
import sounddevice as sd

from speechd.preprocessing import AGC, Pipeline, VoiceActivityDetector

logger = logging.getLogger(__name__)


def run_preview():
    pipeline = Pipeline(
        AGC(),
        VoiceActivityDetector(),
    )

    frames: list[np.ndarray] = []
    recording = True

    def on_audio(indata, _frames, _time, _status):
        if recording:
            frames.append(indata.copy().flatten())

    def on_signal(_signum, _frame):
        nonlocal recording
        recording = False

    signal.signal(signal.SIGINT, on_signal)

    stream = sd.InputStream(
        samplerate=16000,
        channels=1,
        dtype=np.int16,
        callback=on_audio,
    )

    logger.info("Recording... Press Ctrl+C to stop")
    stream.start()

    while recording:
        sd.sleep(100)

    stream.stop()
    stream.close()

    if not frames:
        logger.info("No audio recorded")
        return

    audio_raw = np.concatenate(frames)
    duration_raw = len(audio_raw) / 16000
    logger.info(f"Recorded {duration_raw:.1f}s of audio")

    logger.info("Applying AGC + VAD preprocessing...")
    audio_clean = pipeline.process(audio_raw)

    if len(audio_clean) == 0:
        logger.info("No speech detected after VAD")
        return

    duration_clean = len(audio_clean) / 16000
    logger.info(f"Playing back {duration_clean:.1f}s of processed audio...")

    sd.play(audio_clean, 16000)
    sd.wait()
    logger.info("Done")
