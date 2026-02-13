"""Byte Pair Encoder (BPE) implementation."""

import gzip
import json
import logging
from collections import Counter, OrderedDict
from dataclasses import dataclass
from itertools import chain
from math import ceil
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

Symbol = str

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Word:
    """Represents a word composed of symbols."""

    symbols: tuple[Symbol, ...]

    def merge_pairs(self, rules: list[Bigram]) -> Word:
        """Merge adjacent pairs of symbols according to the given rules.

        Args:
            rules: A list of bigram rules to apply.

        Returns:
            A new Word object with the merged symbols.
        """
        symbols = list(self.symbols)
        for rule in rules:
            new_symbols: list[Symbol] = []
            index = 0
            while index < len(symbols):
                if (
                    index + 1 < len(symbols)
                    and symbols[index] == rule.left
                    and symbols[index + 1] == rule.right
                ):
                    new_symbols.append(rule.merged)
                    index += 2  # Skip the consumed pair
                else:
                    new_symbols.append(symbols[index])
                    index += 1
            symbols = new_symbols
        return Word(tuple(symbols))

    def replace_missing_symbols(
        self, vocab: list[Symbol], missing_token: Symbol
    ) -> Word:
        """Replace unknown symbols with the missing token.

        Args:
            vocab: A list of known symbols.
            missing_token: The token to replace unknown symbols with.

        Returns:
            A new Word object with the replaced symbols.
        """
        return Word(
            tuple(
                missing_token if symbol not in vocab else symbol
                for symbol in self.symbols
            )
        )

    def count_pairs(self) -> Counter[Bigram]:
        """Count the occurrences of adjacent pairs of symbols in the word."""
        counts = Counter()
        for index in range(len(self) - 1):
            pair = Bigram(self[index], self[index + 1])
            counts[pair] += 1
        return counts

    def __str__(self) -> str:
        """Return the string representation of the word."""
        return "".join(self.symbols)

    def __iter__(self) -> Iterator:
        """Return an iterator over the symbols in the word."""
        return iter(self.symbols)

    def __len__(self) -> int:
        """Return the number of symbols in the word."""
        return len(self.symbols)

    def __getitem__(self, index: int) -> Symbol:
        """Return the symbol at the given index."""
        return self.symbols[index]

    @staticmethod
    def from_str(word: str, end_token: str = "</w>") -> Word:  # noqa: S107
        """Create a Word object from a string."""
        return Word(symbols=(*tuple(word), end_token))


class Bigram(NamedTuple):
    """A bigram is a pair of symbols."""

    left: Symbol
    right: Symbol

    @property
    def merged(self) -> Symbol:
        """Return the merged symbol."""
        return self.left + self.right


class BytePairEncoder:
    """Byte Pair Encoder (BPE) is a subword tokenization algorithm."""

    def __init__(
        self,
        vocab: list[Symbol] | None = None,
        rules: list[Bigram] | None = None,
        end_token: str = "</w>",  # noqa: S107
        unknown_token: str = "<unk>",  # noqa: S107
    ) -> None:
        """Initialize the BytePairEncoder.

        Args:
            vocab: The vocabulary of symbols.
            rules: The rules for merging adjacent symbols.
            end_token: The token to use for end of word.
            unknown_token: The token to use for unknown symbols.
        """
        self.vocab = [] if vocab is None else vocab
        self._token_to_id_cache: dict[Symbol, int] = {}
        self.rules = [] if rules is None else rules
        self.unknown_token = unknown_token
        self.end_token = end_token
        self._merge_cache: OrderedDict[tuple[Symbol, ...], Word] = OrderedDict()
        self._merge_cache_size: int = 0
        self._merge_eviction_count: int = 0
        self._merge_eviction_threshold: int = 10

        if vocab is not None and unknown_token not in vocab:
            logger.warning(
                "Initial vocabulary does not contain the unknown token. "
                "Appending the missing token '%s'.",
                unknown_token,
            )
            self.vocab.append(unknown_token)

    @property
    def _merge_cache_frozen(self) -> bool:
        return (
            self._merge_cache_size == 0
            or self._merge_eviction_count >= self._merge_eviction_threshold
        )

    def set_merge_cache(self, size: int, eviction_threshold: int = 10) -> None:
        """Configure the merge cache for encoding optimization.

        Args:
            size: Maximum number of words to cache (0 to disable).
            eviction_threshold: Max evictions before cache becomes fixed.
        """
        self._merge_cache_size = size
        self._merge_eviction_threshold = eviction_threshold
        self._merge_cache.clear()
        self._merge_eviction_count = 0

    def _merge_cache_get(self, word: Word) -> Word | None:
        key = word.symbols
        if key in self._merge_cache:
            if not self._merge_cache_frozen:
                self._merge_cache.move_to_end(key)
            return self._merge_cache[key]
        return None

    def _merge_cache_put(self, word: Word, merged: Word) -> None:
        if self._merge_cache_frozen:
            return
        key = word.symbols
        if key in self._merge_cache:
            return
        if len(self._merge_cache) >= self._merge_cache_size:
            self._merge_cache_evict()
            if self._merge_cache_frozen:
                return
        self._merge_cache[key] = merged

    def _merge_cache_evict(self) -> None:
        evict_count = ceil(self._merge_cache_size * 0.05)
        for _ in range(evict_count):
            if self._merge_cache:
                self._merge_cache.popitem(last=False)
        self._merge_eviction_count += 1

    def __len__(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)

    def id_to_token(self, token_id: int) -> Symbol:
        """Return the token with the given id."""
        return self.vocab[token_id]

    def token_to_id(self, token: Symbol) -> int:
        """Return the id of the given token."""
        if len(self.vocab) != len(self._token_to_id_cache):
            self._token_to_id_cache = {v: i for i, v in enumerate(self.vocab)}
        return self._token_to_id_cache[token]

    def tokenize(self, text: str) -> list[Symbol]:
        """Convert a text string into a list of tokens using Byte Pair Encoding.

        Args:
            text: The input text string to be encoded.
            unknown_token: The token to use for unknown symbols.

        Returns:
            A list of tokens representing the text.
        """
        # Preprocess the text into a list of symbols
        words = self.pre_tokenize(text)
        words = self.merge_symbols(words)
        words = [
            word.replace_missing_symbols(self.vocab, self.unknown_token)
            for word in words
        ]
        return [symbol for word in words for symbol in word.symbols]

    def encode(self, text: str) -> list[int]:
        """Encode a text string into a list of token ids using Byte Pair Encoding.

        Args:
            text: The input text string to be encoded.

        Returns:
            A list of token indices representing the text.
        """
        # Preprocess the text into a list of symbols
        tokens = self.tokenize(text)
        return [self.token_to_id(token) for token in tokens]

    def decode(self, token_ids: list[int]) -> str:
        """Decode a list of token ids into a text string using Byte Pair Encoding.

        Args:
            token_ids: A list of token indices representing the text.

        Returns:
            The decoded text string.
        """
        tokens = [self.id_to_token(token_id) for token_id in token_ids]
        return "".join([token.replace(self.end_token, " ") for token in tokens])

    def train(
        self, corpus: Iterable[str], max_vocab: int, max_iter: int = 100000
    ) -> None:
        """Learn new vocabulary and merge rules from the corpus.

        Args:
            corpus: A text corpus to train the vocabulary and merge rules.
            max_vocab: Continue training till the vocabulary size reaches max_vocab.
            max_iter: Termination condition when max_vocab is not reached.
        """
        word_counts: Counter[Word] = Counter()
        for text_chunk in corpus:
            word_counts.update(self.pre_tokenize(text_chunk))

        if len(self.vocab) == 0:
            self.vocab = [
                self.unknown_token,
                *BytePairEncoder.get_corpus_symbols(set(word_counts.keys())),
            ]
            logger.info(
                "Initialized vocabulary with %d (%.2f%%) symbols",
                len(self.vocab),
                len(self.vocab) / max_vocab * 100,
            )

        if len(self.rules) > 0:
            word_counts = Counter(
                {
                    word.merge_pairs(self.rules): count
                    for word, count in word_counts.items()
                }
            )

        pair_counts = BytePairEncoder.count_bigrams(word_counts)

        for _ in range(max_iter):
            if len(self.vocab) >= max_vocab:
                return
            word_counts, pair_counts = self._train_step(word_counts, pair_counts)
            complete_percent = len(self.vocab) / max_vocab * 100
            if complete_percent % 5 == 0:
                logger.info("Training progress: %.2f%%", complete_percent)

        logger.warning(
            "Training reached maximum iterations without "
            "reaching the desired vocabulary size. "
            "Current vocabulary size: %d, desired size: %d",
            len(self.vocab),
            max_vocab,
        )

    def _train_step(
        self, word_counts: Counter[Word], pair_counts: Counter[Bigram]
    ) -> tuple[Counter[Word], Counter[Bigram]]:
        """Learn a new merge rule from the corpus.

        Args:
            word_counts: Counter of unique words with their frequencies (MUTATED).
            pair_counts: Counter of bigram frequencies across all words (MUTATED).

        Returns:
            The same word_counts and pair_counters objects passed in, modified in-place.
        """
        pair = pair_counts.most_common(1)[0][0]
        self.rules.append(pair)
        self.vocab.append(pair.merged)

        changed_words: set[tuple[Word, Word]] = set()
        for word in word_counts:
            merged_word = word.merge_pairs([pair])
            if merged_word != word:
                changed_words.add((word, merged_word))

        for original_word, merged_word in changed_words:
            count = word_counts[original_word]
            for p, cnt in original_word.count_pairs().items():
                BytePairEncoder._update_pair_count(pair_counts, p, -cnt * count)
            for p, cnt in merged_word.count_pairs().items():
                BytePairEncoder._update_pair_count(pair_counts, p, cnt * count)
            del word_counts[original_word]
            word_counts[merged_word] += count

        return word_counts, pair_counts

    @staticmethod
    def _update_pair_count(
        pair_counts: Counter[Bigram], pair: Bigram, delta: int
    ) -> None:
        """Update pair count, removing entry if count drops to zero or below.

        Args:
            pair_counts: Counter of bigram frequencies (MUTATED).
            pair: The bigram to update.
            delta: The change in count (positive or negative).
        """
        new_count = pair_counts[pair] + delta
        if new_count <= 0:
            del pair_counts[pair]
        else:
            pair_counts[pair] = new_count

    @staticmethod
    def get_corpus_symbols(words: set[Word]) -> set[Symbol]:
        """Get all symbols from a corpus.

        Args:
            words: A set of unique words.

        Returns:
            All unique symbols in the corpus.
        """
        corpus_symbols = chain.from_iterable([word.symbols for word in words])
        return set(corpus_symbols)

    @staticmethod
    def find_most_frequent_pair(word_counts: Counter[Word]) -> Bigram:
        """Find the most frequent pair of adjacent symbols in a corpus.

        Args:
            word_counts: Counter of unique words with their frequencies.

        Returns:
            The most frequent pair of adjacent symbols.
        """
        pair_counts = BytePairEncoder.count_bigrams(word_counts)
        return pair_counts.most_common(1)[0][0]

    def pre_tokenize(self, text_chunk: str) -> list[Word]:
        """Split a text chunk into words.

        Args:
            text_chunk: A text chunk to split into words.
        """
        return [Word.from_str(w, self.end_token) for w in text_chunk.split()]

    def merge_symbols(self, words: list[Word]) -> list[Word]:
        """Merge adjacent symbols in a word list according to the given rules.

        Args:
            words: A list of words to merge symbols in.
        """
        result = []
        for word in words:
            cached = self._merge_cache_get(word)
            if cached is not None:
                result.append(cached)
            else:
                merged = word.merge_pairs(self.rules)
                self._merge_cache_put(word, merged)
                result.append(merged)
        return result

    @staticmethod
    def count_words(words: list[Word]) -> Counter[Word]:
        """Count the number of occurrences of each word.

        Args:
            words: A list of words to count.
        """
        return Counter(words)

    @staticmethod
    def count_bigrams(word_counts: Counter[Word]) -> Counter[Bigram]:
        """Count the number of adjacent symbol pair occurrences in words.

        Args:
            word_counts: A counter for the number of occurrences of each word.
        """
        counts = Counter()
        for word, word_occurences in word_counts.items():
            pair_counts_in_word = word.count_pairs()
            new_pair_counts = {
                k: v * word_occurences for k, v in pair_counts_in_word.items()
            }
            for pair, pair_occurence in new_pair_counts.items():
                counts[pair] += pair_occurence
        return counts


class BytePairEncoderJSONRepository:
    """JSON repository for persisting and loading BytePairEncoder."""

    @staticmethod
    def save(encoder: BytePairEncoder, path: Path) -> None:
        """Save encoder to compressed JSON file.

        Args:
            encoder: The BytePairEncoder to serialize.
            path: Path to the output compressed JSON file.
        """
        vocab_to_id = {token: idx for idx, token in enumerate(encoder.vocab)}
        data = {
            "vocab": encoder.vocab,
            "rules": [
                [vocab_to_id[r.left], vocab_to_id[r.right]] for r in encoder.rules
            ],
            "end_token": encoder.end_token,
            "unknown_token": encoder.unknown_token,
        }
        json_bytes = json.dumps(data, separators=(",", ":")).encode("utf-8")
        compressed = gzip.compress(json_bytes)
        path.write_bytes(compressed)

    @staticmethod
    def load(path: Path) -> BytePairEncoder:
        """Load encoder from compressed JSON file.

        Args:
            path: Path to the input compressed JSON file.

        Returns:
            A deserialized BytePairEncoder instance.
        """
        compressed = path.read_bytes()
        json_bytes = gzip.decompress(compressed)
        data = json.loads(json_bytes.decode("utf-8"))
        vocab = data["vocab"]
        rules = [Bigram(left=vocab[r[0]], right=vocab[r[1]]) for r in data["rules"]]
        return BytePairEncoder(
            vocab=vocab,
            rules=rules,
            end_token=data["end_token"],
            unknown_token=data["unknown_token"],
        )
