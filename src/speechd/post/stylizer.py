import itertools
import logging
from difflib import SequenceMatcher
from pathlib import Path

import yaml
from groq import Groq
from pydantic import BaseModel

from speechd.post import register

logger = logging.getLogger(__name__)

_ALLOWED_ARGS = frozenset(
    (
        "model",
        "temperature",
        "top_p",
        "reasoning_effort",
        "max_tokens",
    )
)

_DEFAULT_ARGS = {
    "model": "qwen/qwen3-32b",
    "reasoning_effort": "none",
}


class Prompt:
    def __init__(self, args: dict, messages: list[dict]):
        self.args = args
        self.messages = messages

    @classmethod
    def parse(cls, md: str) -> "Prompt":
        segments = list(map(str.strip, md.split("\n---\n")))

        if segments[0].startswith("---\n"):
            frontmatter = segments[0][4:].strip()
            segments = segments[1:]
            args = yaml.safe_load(frontmatter) or {}
        else:
            args = {}

        if not _ALLOWED_ARGS.issuperset(args.keys()):
            unknown = args.keys() - _ALLOWED_ARGS
            raise ValueError(f"Forbidden prompt args: {unknown}")

        messages = [{"role": "system", "content": segments[0]}]

        for user, assistant in itertools.batched(segments[1:], n=2, strict=True):
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})

        return cls(args=args, messages=messages)


@register("stylizer")
class Stylizer:
    class Config(BaseModel):
        api_key: str
        prompt_file: Path
        similarity_threshold: float = 0.15
        max_retries: int = 3

    def __init__(self, config: Config):
        self.config = config
        self._client = Groq(api_key=config.api_key)
        self._prompt = Prompt.parse(config.prompt_file.read_text())

    def _process_once(self, raw: str) -> str:
        kwargs = _DEFAULT_ARGS | self._prompt.args

        messages = list(
            itertools.chain(
                self._prompt.messages,
                [
                    {
                        "role": "user",
                        "content": raw,
                    }
                ],
            )
        )

        completion = self._client.chat.completions.create(
            messages=messages,
            stream=False,
            **kwargs,
        )

        return completion.choices[0].message.content

    def process(self, text: str) -> str:
        for attempt in range(self.config.max_retries):
            processed = self._process_once(text)
            similarity = SequenceMatcher(None, text, processed).ratio()

            if similarity > self.config.similarity_threshold:
                return processed

            logger.info(
                "Attempt %d: result too different (sim=%.2f), retrying",
                attempt + 1,
                similarity,
            )

        logger.warning("All retries exhausted, returning original text")
        return text
