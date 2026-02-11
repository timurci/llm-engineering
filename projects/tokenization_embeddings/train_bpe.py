"""Train a BytePairEncoder from scratch using a HuggingFace dataset."""

import logging
from pathlib import Path

from datasets import load_dataset

from llm_engineering.tokenizer.bpe import BytePairEncoder, BytePairEncoderJSONRepository

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
MAX_VOCAB_SIZE = 2000
OUTPUT_PATH = Path("models/bpe_tokenizer.json")

logging.basicConfig(level=logging.INFO)


def main() -> None:
    """Train a BPE tokenizer and save it to disk."""
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    corpus = (sample["text"] for sample in dataset if sample["text"])  # type: ignore[index]

    encoder = BytePairEncoder()
    encoder.train(corpus, max_vocab=MAX_VOCAB_SIZE)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BytePairEncoderJSONRepository.save(encoder, OUTPUT_PATH)
    print(f"Saved tokenizer to {OUTPUT_PATH}")  # noqa: T201

    tokens = encoder.tokenize("Hello world!")
    print(f"Tokens: {tokens}")  # noqa: T201


if __name__ == "__main__":
    main()
