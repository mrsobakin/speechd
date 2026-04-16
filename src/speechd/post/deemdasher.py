import re

from pydantic import BaseModel

from speechd.post import register

_SYMBOLS = str.maketrans({
    "—": "-",
    "–": "-",
    "…": "...",
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "«": '"',
    "»": '"',
    "\u00A0": " ",  # non-breaking space
    "\u2009": " ",  # thin space
    "\u202F": " ",  # narrow no-break space
    "\u200B": "",   # zero-width space
    "\u00AD": "",   # soft hyphen
})


@register("deemdasher")
class DeEmdasher:
    class Config(BaseModel):
        doubledash: bool = False

    def __init__(self, config: Config | None = None):
        self.config = config or self.Config()

    def process(self, text: str) -> str:
        text = text.translate(_SYMBOLS)

        if self.config.doubledash:
            text = re.sub(r"(\s)-(\s)", r"\1--\2", text)

        return text
