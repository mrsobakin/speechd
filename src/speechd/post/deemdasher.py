import re
from dataclasses import dataclass

_SYMBOLS = str.maketrans({
    "—": "-",
    "–": "-",
    "…": "...",
    "‘": "'",
    "’": "'",
    "“": "\"",
    "”": "\"",
    "„": "\"",
    "«": "\"",
    "»": "\"",
    "\u00A0": " ",  # non-breaking space
    "\u2009": " ",  # thin space
    "\u202F": " ",  # narrow no-break space
    "\u200B": "",   # zero-width space
    "\u00AD": "",   # soft hyphen
})

@dataclass
class DeEmdasher:
    doubledash: bool = False

    def process(self, text: str) -> str:
        text = text.translate(_SYMBOLS)

        if self.doubledash:
            text = re.sub(r"(\s)-(\s)", r"\1--\2", text)

        return text
