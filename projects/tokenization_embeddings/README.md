# Tokenization & Embeddings

Implementation of byte-pair encoding (BPE) tokenization and learned embeddings.

## BytePairEncoder

BPE is a subword tokenization algorithm that iteratively merges the most frequent adjacent symbol pairs:
1. Start with a vocabulary of individual characters
2. Count all adjacent symbol pairs in the corpus
3. Merge the most frequent pair into a new symbol
4. Repeat until the target vocabulary size is reached

This produces subword units that balance between character-level and word-level granularity.

## BigramEmbeddingModel

A simple neural language model for next-token prediction:
- **Architecture**: Embedding layer (maps token IDs to embedding_dim vectors) + Linear layer (embedding_dim → vocab_size)
- **Training**: Given a token, predict the next token. Cross-entropy loss over the vocabulary.
- **Purpose**: Learn dense embeddings where semantically similar words cluster together.

## Scripts

| Script | Purpose | Run | Input | Output |
|--------|---------|-----|-------|--------|
| `train_bpe.py` | Train a BPE tokenizer | `uv run train_bpe.py` | WikiText-2 corpus | `models/bpe_tokenizer.gzip` |
| `visualize_bpe.py` | Visualize tokenization | `uv run visualize_bpe.py -t "text"` or `uv run visualize_bpe.py -i` | Trained tokenizer + text | Colored terminal output |
| `train_embedding.py` | Train embedding model | `uv run train_embedding.py [--tokenizer-path PATH]` | Trained tokenizer + WikiText-2 | `models/embedding_model.pt` |
| `visualize_embeddings.py` | Compare one-hot vs learned embeddings | `uv run visualize_embeddings.py` | Trained tokenizer + trained model + `word_categories.yaml` | `outputs/embedding_visualizations/*.png` |

Use `-h` or `--help` on any script for additional options.

> **Note**: Use `uv run mlflow ui` to track `train_embedding.py` experiments.
