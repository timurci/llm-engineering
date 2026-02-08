"""Byte Pair Encoder (BPE) implementation."""

import logging
from collections import Counter
from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

Symbol = str

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Word:
    """Represents a word composed of symbols."""

    symbols: list[Symbol]

    def merge_pairs(self, rules: list[Bigram]) -> Word:
        """Merge adjacent pairs of symbols according to the given rules.

        Args:
            rules: A list of bigram rules to apply.

        Returns:
            A new Word object with the merged symbols.
        """
        word = Word(self.symbols.copy())
        for rule in rules:
            if rule.left not in word.symbols:
                continue
            left_index = word.symbols.index(rule.left)
            right_index = left_index + 1
            if (
                right_index < len(word.symbols)
                and word.symbols[right_index] == rule.right
            ):
                left_hand = word.symbols[:left_index]
                right_hand = word.symbols[(right_index + 1) :]
                word = Word([*left_hand, rule.merged, *right_hand])
        return word

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
            [
                missing_token if symbol not in vocab else symbol
                for symbol in self.symbols
            ]
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
        return Word(symbols=[*list(word), end_token])


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
        self.rules = [] if rules is None else rules
        self.unknown_token = unknown_token
        self.end_token = end_token

    def encode(self, text: str) -> list[str]:
        """Encode a text string into a list of tokens using Byte Pair Encoding.

        Args:
            text: The input text string to be encoded.
            unknown_token: The token to use for unknown symbols.

        Returns:
            A list of tokens representing the encoded text.
        """
        # Preprocess the text into a list of symbols
        words = self.pre_tokenize(text)
        words = self.merge_symbols(words)
        words = [
            word.replace_missing_symbols(self.vocab, self.unknown_token)
            for word in words
        ]
        return [str(word) for word in words]

    def train(
        self, corpus: Iterable[str], max_vocab: int, max_iter: int = 100000
    ) -> None:
        """Learn new vocabulary and merge rules from the corpus.

        Args:
            corpus: A text corpus to train the vocabulary and merge rules.
            max_vocab: Continue training till the vocabulary size reaches max_vocab.
            max_iter: Termination condition when max_vocab is not reached.
        """
        # Flatten the corpus into a list of words
        # The initial symbols are just characters.
        corpus_words = list(
            chain.from_iterable(
                [self.pre_tokenize(text_chunk) for text_chunk in corpus]
            )
        )

        # Initialize vocabulary if it's empty with all characters in the corpus.
        if len(self.vocab) == 0:
            self.vocab = list(BytePairEncoder.get_corpus_symbols(corpus_words))

        # If there are existing rules, merge corpus symbols accordingly.
        if len(self.rules) > 0:
            corpus_words = self.merge_symbols(corpus_words)

        for _ in range(max_iter):
            if len(self.vocab) >= max_vocab:
                return
            corpus_words = self._train_step(corpus_words)

        logger.warning(
            "Training reached maximum iterations without "
            "reaching the desired vocabulary size. "
            "Current vocabulary size: %d, desired size: %d",
            len(self.vocab),
            max_vocab,
        )

    def _train_step(self, corpus: list[Word]) -> list[Word]:
        """Learn a new merge rule from the corpus.

        Args:
            corpus: The list of all words in the corpus.

        Returns:
            The updated corpus with the new merge rule applied.
        """
        pair = BytePairEncoder.find_most_frequent_pair(corpus)
        self.rules.append(pair)
        self.vocab.append(pair.merged)

        return self.merge_symbols(corpus)

    @staticmethod
    def get_corpus_symbols(corpus: list[Word]) -> set[Symbol]:
        """Get all symbols from a corpus.

        Args:
            corpus: The list of all words in the whole corpus.

        Returns:
            All unique symbols in the corpus.
        """
        corpus_symbols = chain.from_iterable([word.symbols for word in corpus])
        return set(corpus_symbols)

    @staticmethod
    def find_most_frequent_pair(corpus: list[Word]) -> Bigram:
        """Find the most frequent pair of adjacent symbols in a corpus.

        Args:
            corpus: The list of words in the whole corpus.

        Returns:
            The most frequent pair of adjacent symbols.
        """
        word_counts = BytePairEncoder.count_words(corpus)
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
        return [word.merge_pairs(self.rules) for word in words]

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
