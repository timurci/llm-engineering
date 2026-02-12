"""Train an embedding model using a pretrained BPE tokenizer."""

import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from llm_engineering.core.trackers import MLflowTracker, StandardLoggingTracker
from llm_engineering.embedding.dataset import BigramEmbeddingDataset
from llm_engineering.embedding.model import BigramEmbeddingModel
from llm_engineering.embedding.training import BigramEmbeddingTrainer
from llm_engineering.tokenizer.bpe import BytePairEncoder, BytePairEncoderJSONRepository

DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
EMBEDDING_DIM = 128
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
LOGGING_INTERVAL = EPOCHS // 10
TOKENIZER_PATH = Path("models/bpe_tokenizer.gzip")
OUTPUT_PATH = Path("models/embedding_model.pt")

logging.basicConfig(
    format="%(asctime)s | %(levelname)s:%(name)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Train an embedding model with a BPE tokenizer")
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=TOKENIZER_PATH,
        help=f"Path to pretrained BPE tokenizer (default: {TOKENIZER_PATH})",
    )
    return parser


def encode_dataset(dataset: Dataset, tokenizer: BytePairEncoder) -> list[int]:
    """Encode a dataset using a BPE tokenizer.

    Args:
        dataset: The huggingface dataset to encode.
        tokenizer: The BPE tokenizer to use.

    Returns:
        A list of token IDs.
    """
    all_tokens: list[int] = []
    for sample in tqdm(dataset, desc="Encoding", leave=False):
        if sample["text"]:
            all_tokens.extend(tokenizer.encode(sample["text"]))
    return all_tokens


def main() -> None:
    """Train an embedding model with a BPE tokenizer."""
    args = _build_parser().parse_args()

    logger.info("Loading tokenizer from %s", args.tokenizer_path)
    tokenizer = BytePairEncoderJSONRepository.load(args.tokenizer_path)
    vocab_size = len(tokenizer)
    logger.info("Vocabulary size: %d", vocab_size)

    logger.info("Loading dataset %s/%s", DATASET_NAME, DATASET_CONFIG)
    train_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="train")
    val_dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split="validation")

    logger.info("Encoding training data")
    train_tokens = encode_dataset(train_dataset, tokenizer)
    logger.info("Training tokens: %d", len(train_tokens))

    logger.info("Encoding validation data")
    val_tokens = encode_dataset(val_dataset, tokenizer)
    logger.info("Validation tokens: %d", len(val_tokens))

    train_data = BigramEmbeddingDataset(train_tokens)
    val_data = BigramEmbeddingDataset(val_tokens)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = BigramEmbeddingModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    trainer = BigramEmbeddingTrainer(model=model, optimizer=optimizer)

    params = {
        "vocab_size": vocab_size,
        "embedding_dim": EMBEDDING_DIM,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "optimizer": str(optimizer),
        "dataset": f"{DATASET_NAME}/{DATASET_CONFIG}",
        "tokenizer_path": str(args.tokenizer_path),
    }

    with (
        StandardLoggingTracker(
            logger=logger, logging_interval=LOGGING_INTERVAL, experiment_steps=EPOCHS
        ) as logging_tracker,
        MLflowTracker(
            experiment_name="embedding_training",
            run_name=f"bigram_{DATASET_CONFIG}",
        ) as mlflow_tracker,
    ):
        logging_tracker.log_params(params)
        mlflow_tracker.log_params(params)

        trainer.train(
            train_loader=train_loader,
            device=device,
            epochs=EPOCHS,
            val_loader=val_loader,
            experiment_trackers=[logging_tracker, mlflow_tracker],
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), OUTPUT_PATH)
    logger.info("Saved model to %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
