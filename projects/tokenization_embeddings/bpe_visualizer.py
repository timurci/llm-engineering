"""Text visualization script for Byte Pair Encoding."""

import logging
from argparse import ArgumentParser
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

from llm_engineering.tokenizer.bpe import BytePairEncoder, BytePairEncoderJSONRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EscapeSequence:
    """Escape sequence for text formatting."""

    start: str
    end: str = "\033[0m"


@dataclass(frozen=True)
class VisualConfig:
    """Configuration for text visualization."""

    colors: cycle[EscapeSequence]
    override_tokens: dict[str, str]


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="CLI Text Visualizer for BPE Tokenization")
    parser.add_argument("-t", "--text", default=None, help="Text to visualize")
    parser.add_argument(
        "-i", "--interactive", action="store_true", help="Enable interactive mode"
    )
    parser.add_argument(
        "-e", "--encoder", default="bpe_encoder.json", help="Path to BPE encoder file"
    )
    return parser


def enrich_text(text: str, encoder: BytePairEncoder, config: VisualConfig) -> str:
    """Enriches the text with color and overrides tokens.

    Args:
        text: Text to enrich with colors.
        encoder: BytePairEncoder instance.
        config: VisualConfig instance.
    """
    encoded_text = encoder.tokenize(text)
    colors = config.colors
    enriched_text = ""

    for token in encoded_text:
        color = next(colors)
        enriched_token = color.start + token + color.end
        for exception in config.override_tokens:
            if exception in token:
                enriched_token = enriched_token.replace(
                    exception, config.override_tokens[exception]
                )
        enriched_text += enriched_token

    return enriched_text


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    encoder = BytePairEncoderJSONRepository.load(Path(args.encoder))
    config = VisualConfig(
        colors=cycle(
            [
                EscapeSequence("\033[41m"),
                EscapeSequence("\033[42m"),
                EscapeSequence("\033[44m"),
                EscapeSequence("\033[43m"),
                EscapeSequence("\033[45m"),
            ]
        ),
        override_tokens={encoder.end_token: " ", encoder.unknown_token: "?"},
    )

    if args.interactive:
        logger.info("Press Ctrl+D or type 'exit' to exit")
        while text := input(""):
            if text == "exit":
                break
            print(enrich_text(text, encoder, config))  # noqa: T201
    else:
        print(enrich_text(args.text, encoder, config))  # noqa: T201
