"""Visualize and compare one-hot encoding vs learned embeddings."""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances

from llm_engineering.embedding.model import BigramEmbeddingModel
from llm_engineering.tokenizer.bpe import BytePairEncoder, BytePairEncoderJSONRepository

TOKENIZER_PATH = Path("models/bpe_tokenizer.gzip")
EMBEDDING_MODEL_PATH = Path("models/embedding_model.pt")
EMBEDDING_DIM = 32
MIN_WORDS_FOR_VISUALIZATION = 5
YAML_PATH = Path("./word_categories.yaml")


@dataclass
class VisualizationData:
    """Container for visualization data."""

    one_hot: np.ndarray
    embeddings: np.ndarray
    words: list[str]
    categories: list[str]
    category_colors: dict[str, str]


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        description="Visualize embeddings with heatmaps and dimensionality reduction"
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=TOKENIZER_PATH,
        help=f"Path to pretrained BPE tokenizer (default: {TOKENIZER_PATH})",
    )
    parser.add_argument(
        "--embedding-model-path",
        type=Path,
        default=EMBEDDING_MODEL_PATH,
        help=f"Path to trained embedding model (default: {EMBEDDING_MODEL_PATH})",
    )
    parser.add_argument(
        "--yaml-path",
        type=Path,
        default=YAML_PATH,
        help=f"Path to YAML config with word categories (default: {YAML_PATH})",
    )
    return parser


def load_yaml_config(yaml_path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Load word categories and colors from YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Tuple of (word_categories, category_colors) dictionaries.
    """
    with yaml_path.open("r") as f:
        config = yaml.safe_load(f)

    word_categories = config["word_categories"]
    category_colors = config["category_colors"]

    return word_categories, category_colors


def load_models(
    tokenizer_path: Path, embedding_model_path: Path
) -> tuple[BytePairEncoder, BigramEmbeddingModel]:
    """Load tokenizer and embedding model from disk.

    Args:
        tokenizer_path: Path to the BPE tokenizer file.
        embedding_model_path: Path to the embedding model file.

    Returns:
        Tuple of (tokenizer, model).
    """
    tokenizer = BytePairEncoderJSONRepository.load(tokenizer_path)
    vocab_size = len(tokenizer)

    model = BigramEmbeddingModel(vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM)
    model.load_state_dict(
        torch.load(embedding_model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    return tokenizer, model


def filter_single_token_words(
    tokenizer: BytePairEncoder, word_categories: dict[str, list[str]]
) -> tuple[list[str], list[str]]:
    """Filter words that tokenize to a single token.

    Args:
        tokenizer: The BPE tokenizer.
        word_categories: Dictionary mapping category names to word lists.

    Returns:
        Tuple of (words, categories) lists for single-token words.
    """
    single_token_words = []
    categories = []

    for category, words in word_categories.items():
        for word in words:
            tokens = tokenizer.tokenize(word)
            if len(tokens) == 1:
                single_token_words.append(word)
                categories.append(category)

    return single_token_words, categories


def get_one_hot_encodings(words: list[str], tokenizer: BytePairEncoder) -> np.ndarray:
    """Generate one-hot encodings for a list of words.

    Args:
        words: List of words to encode.
        tokenizer: The BPE tokenizer.

    Returns:
        One-hot encoding matrix of shape (num_words, vocab_size).
    """
    vocab_size = len(tokenizer)
    one_hot = np.zeros((len(words), vocab_size))

    for i, word in enumerate(words):
        token_id = tokenizer.token_to_id(tokenizer.tokenize(word)[0])
        one_hot[i, token_id] = 1.0

    return one_hot


def get_learned_embeddings(
    words: list[str], tokenizer: BytePairEncoder, model: BigramEmbeddingModel
) -> np.ndarray:
    """Extract learned embeddings for a list of words.

    Args:
        words: List of words to embed.
        tokenizer: The BPE tokenizer.
        model: The embedding model.

    Returns:
        Embedding matrix of shape (num_words, embedding_dim).
    """
    token_ids = [tokenizer.token_to_id(tokenizer.tokenize(word)[0]) for word in words]
    token_tensor = torch.tensor(token_ids, dtype=torch.long)

    with torch.inference_mode():
        embeddings = model.encoder(token_tensor)

    return embeddings.numpy()


def plot_heatmaps(
    data: VisualizationData,
    output_path: Path,
) -> None:
    """Create side-by-side heatmaps of cosine distances with category color bars.

    Args:
        data: Visualization data container.
        output_path: Path to save the output figure.
    """
    one_hot_dist = cosine_distances(data.one_hot)
    embed_dist = cosine_distances(data.embeddings)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    heatmap_configs = [
        (one_hot_dist, "One-Hot Encoding\n(Pairwise Cosine Distance)"),
        (embed_dist, "Learned Embeddings\n(Pairwise Cosine Distance)"),
    ]

    for ax, (dist_matrix, title) in zip(axes, heatmap_configs, strict=True):
        sns.heatmap(
            dist_matrix,
            ax=ax,
            xticklabels=data.words,
            yticklabels=data.words,
            cmap="viridis",
            annot=False,
            fmt=".2f",
            square=True,
            cbar_kws={"label": "Cosine Distance", "shrink": 0.8},
        )
        ax.set_title(title, fontsize=12, pad=20)

        for i, cat in enumerate(data.categories):
            color = data.category_colors[cat]
            ax.axhspan(i - 0.5, i + 0.5, xmin=-0.04, xmax=0, color=color, clip_on=False)
            ax.axvspan(
                i - 0.5, i + 0.5, ymin=1.0, ymax=1.04, color=color, clip_on=False
            )

    legend_handles = [
        mpatches.Rectangle((0, 0), 1, 1, facecolor=color, label=cat)
        for cat, color in data.category_colors.items()
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.02),
        title="Word Categories",
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved heatmap to {output_path}")  # noqa: T201


def plot_dimensionality_reduction(
    data: VisualizationData,
    output_path: Path,
) -> None:
    """Create side-by-side PCA and t-SNE visualizations.

    Args:
        data: Visualization data container.
        output_path: Path to save the output figure.
    """
    pca_one_hot = PCA(n_components=2).fit_transform(data.one_hot)
    pca_embeddings = PCA(n_components=2).fit_transform(data.embeddings)

    tsne_one_hot = TSNE(
        n_components=2, random_state=42, perplexity=min(5, len(data.words) - 1)
    ).fit_transform(data.one_hot)
    tsne_embeddings = TSNE(
        n_components=2, random_state=42, perplexity=min(5, len(data.words) - 1)
    ).fit_transform(data.embeddings)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    for transformed_data, title, ax in [
        (pca_one_hot, "PCA - One-Hot Encoding", axes[0, 0]),
        (pca_embeddings, "PCA - Learned Embeddings", axes[0, 1]),
        (tsne_one_hot, "t-SNE - One-Hot Encoding", axes[1, 0]),
        (tsne_embeddings, "t-SNE - Learned Embeddings", axes[1, 1]),
    ]:
        for cat in set(data.categories):
            mask = [c == cat for c in data.categories]
            ax.scatter(
                transformed_data[mask, 0],
                transformed_data[mask, 1],
                c=data.category_colors[cat],
                label=cat,
                s=100,
                alpha=0.8,
            )

        for i, word in enumerate(data.words):
            ax.annotate(
                word,
                (transformed_data[i, 0], transformed_data[i, 1]),
                fontsize=9,
                ha="center",
                va="bottom",
            )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.legend(loc="best")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved dimensionality reduction plot to {output_path}")  # noqa: T201


def main() -> None:
    """Run the embedding visualization pipeline."""
    args = _build_parser().parse_args()

    word_categories, category_colors = load_yaml_config(args.yaml_path)
    print(f"Loaded config from {args.yaml_path}")  # noqa: T201

    tokenizer, model = load_models(args.tokenizer_path, args.embedding_model_path)
    print(f"Loaded tokenizer with vocab size: {len(tokenizer)}")  # noqa: T201

    words, categories = filter_single_token_words(tokenizer, word_categories)
    print(f"Found {len(words)} single-token words: {words}")  # noqa: T201

    if len(words) < MIN_WORDS_FOR_VISUALIZATION:
        msg = (
            f"Not enough single-token words for visualization "
            f"(found {len(words)}, need {MIN_WORDS_FOR_VISUALIZATION})"
        )
        raise ValueError(msg)

    one_hot = get_one_hot_encodings(words, tokenizer)
    embeddings = get_learned_embeddings(words, tokenizer, model)

    vis_data = VisualizationData(
        one_hot=one_hot,
        embeddings=embeddings,
        words=words,
        categories=categories,
        category_colors=category_colors,
    )

    output_dir = Path("outputs/embedding_visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_heatmaps(vis_data, output_dir / "heatmap_comparison.png")

    plot_dimensionality_reduction(vis_data, output_dir / "projection_comparison.png")


if __name__ == "__main__":
    main()
