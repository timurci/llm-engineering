"""Tests for Byte Pair Encoder (BPE) implementation."""

from collections import Counter
from typing import TYPE_CHECKING

from llm_engineering.tokenizer.bpe import (
    Bigram,
    BytePairEncoder,
    BytePairEncoderJSONRepository,
    Word,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestWord:
    """Tests for the Word dataclass."""

    def test_from_str(self) -> None:
        """Test creating Word from string with various words and end tokens."""
        assert Word.from_str("apple", "</end>").symbols == (
            "a",
            "p",
            "p",
            "l",
            "e",
            "</end>",
        )
        assert Word.from_str("a", "</w>").symbols == ("a", "</w>")
        assert Word.from_str("", "</w>").symbols == ("</w>",)
        assert Word.from_str("hello", "<END>").symbols == (
            "h",
            "e",
            "l",
            "l",
            "o",
            "<END>",
        )
        assert Word.from_str("world", "$").symbols == ("w", "o", "r", "l", "d", "$")

    def test_count_pairs(self) -> None:
        """Test counting adjacent pairs of symbols with various scenarios."""
        # Single pair
        word = Word(symbols=("un", "help", "ful"))
        pairs = word.count_pairs()
        assert pairs == Counter(
            {
                Bigram("un", "help"): 1,
                Bigram("help", "ful"): 1,
            }
        )

        # Multiple of the same adjacent pairs
        word2 = Word(symbols=("a", "b", "a", "b", "a", "b"))
        pairs2 = word2.count_pairs()
        assert pairs2 == Counter(
            {
                Bigram("a", "b"): 3,
                Bigram("b", "a"): 2,
            }
        )

        # Word with all same adjacent pairs
        word3 = Word(symbols=("x", "x", "x", "x"))
        pairs3 = word3.count_pairs()
        assert pairs3 == Counter({Bigram("x", "x"): 3})

        # Single symbol (no pairs)
        word4 = Word(symbols=("a",))
        pairs4 = word4.count_pairs()
        assert pairs4 == Counter()

        # Empty word
        word5 = Word(symbols=())
        pairs5 = word5.count_pairs()
        assert pairs5 == Counter()

    def test_replace_missing_symbols(self) -> None:
        """Test replacing unknown symbols with missing token."""
        word = Word(symbols=("a", "b", "c", "d", "e"))
        vocab = ["a", "c", "e"]
        missing_token = "<unk>"  # noqa: S105

        result = word.replace_missing_symbols(vocab, missing_token)
        assert result.symbols == ("a", "<unk>", "c", "<unk>", "e")

        # Test with empty vocab (all symbols replaced)
        result2 = word.replace_missing_symbols([], missing_token)
        assert result2.symbols == ("<unk>", "<unk>", "<unk>", "<unk>", "<unk>")

        # Test with all symbols in vocab (no replacements)
        result3 = word.replace_missing_symbols(["a", "b", "c", "d", "e"], missing_token)
        assert result3.symbols == ("a", "b", "c", "d", "e")

    def test_merge_pairs(self) -> None:
        """Test merging adjacent pairs according to rules with various scenarios."""
        word = Word(symbols=tuple("biggest"))

        ruleset1 = [Bigram("s", "t")]
        ruleset2 = [*ruleset1, Bigram("b", "i")]
        ruleset3 = [*ruleset2, Bigram("bi", "g")]

        assert word.merge_pairs(ruleset1).symbols == (*"bigge", "st")
        assert word.merge_pairs(ruleset2).symbols == ("bi", *"gge", "st")
        assert word.merge_pairs(ruleset3).symbols == ("big", "g", "e", "st")

        # Multiple occurrences of same bigram
        word2 = Word(symbols=("a", "b", "a", "b", "a", "b"))
        ruleset4 = [Bigram("a", "b")]
        result = word2.merge_pairs(ruleset4)
        assert result.symbols == ("ab", "ab", "ab")

        # No matching rules
        word3 = Word(symbols=("a", "b", "c"))
        ruleset5 = [Bigram("x", "y")]
        result2 = word3.merge_pairs(ruleset5)
        assert result2.symbols == ("a", "b", "c")

        # Empty ruleset
        result3 = word3.merge_pairs([])
        assert result3.symbols == ("a", "b", "c")


class TestBytePairEncoderStaticMethods:
    """Tests for BytePairEncoder static methods."""

    def test_get_corpus_symbols_empty(self) -> None:
        """Test get_corpus_symbols with empty corpus."""
        assert BytePairEncoder.get_corpus_symbols([]) == set()

    def test_get_corpus_symbols_single_word(self) -> None:
        """Test get_corpus_symbols with single word."""
        corpus = [Word(symbols=("a", "b", "c"))]
        assert BytePairEncoder.get_corpus_symbols(corpus) == {"a", "b", "c"}

    def test_get_corpus_symbols_multiple_words(self) -> None:
        """Test get_corpus_symbols with multiple words and duplicates."""
        corpus = [
            Word(symbols=("a", "b", "c")),
            Word(symbols=("b", "c", "d")),
            Word(symbols=("c", "d", "e")),
        ]
        assert BytePairEncoder.get_corpus_symbols(corpus) == {"a", "b", "c", "d", "e"}

    def test_count_words_empty(self) -> None:
        """Test count_words with empty list."""
        assert BytePairEncoder.count_words([]) == Counter()

    def test_count_words_single(self) -> None:
        """Test count_words with single word."""
        word = Word(symbols=("a", "b"))
        assert BytePairEncoder.count_words([word]) == Counter({word: 1})

    def test_count_words_multiple(self) -> None:
        """Test count_words with multiple occurrences."""
        word1 = Word(symbols=("a", "b"))
        word2 = Word(symbols=("c", "d"))
        words = [word1, word1, word2, word1]
        assert BytePairEncoder.count_words(words) == Counter({word1: 3, word2: 1})

    def test_count_bigrams_simple(self) -> None:
        """Test count_bigrams with simple word counts."""
        word1 = Word(symbols=("a", "b", "c"))
        word2 = Word(symbols=("b", "c", "d"))
        word_counts = Counter({word1: 2, word2: 1})

        result = BytePairEncoder.count_bigrams(word_counts)
        # word1 (count=2): pairs are ("a","b") and ("b","c")
        # word2 (count=1): pairs are ("b","c") and ("c","d")
        expected = Counter(
            {
                Bigram("a", "b"): 2,
                Bigram("b", "c"): 3,  # 2 from word1 + 1 from word2
                Bigram("c", "d"): 1,
            }
        )
        assert result == expected

    def test_find_most_frequent_pair_simple(self) -> None:
        """Test find_most_frequent_pair with clear winner."""
        word1 = Word(symbols=("a", "b", "c"))
        word2 = Word(symbols=("a", "b", "d"))
        corpus = [word1, word2]

        result = BytePairEncoder.find_most_frequent_pair(corpus)
        assert result == Bigram("a", "b")  # Appears in both words

    def test_find_most_frequent_pair_tied(self) -> None:
        """Test find_most_frequent_pair with tied frequencies."""
        word1 = Word(symbols=("a", "b"))
        word2 = Word(symbols=("c", "d"))
        word3 = Word(symbols=("e", "f"))
        corpus = [word1, word2, word3, word1, word2]

        result = BytePairEncoder.find_most_frequent_pair(corpus)
        # Both pairs appear once, most_common(1) returns first encountered
        assert result in {Bigram("a", "b"), Bigram("c", "d")}


class TestBytePairEncoderInstanceMethods:
    """Tests for BytePairEncoder instance methods."""

    def test_pre_tokenize_single_word(self) -> None:
        """Test pre_tokenize with single word."""
        encoder = BytePairEncoder(end_token="</w>")  # noqa: S106
        result = encoder.pre_tokenize("hello")

        assert len(result) == 1
        assert result[0].symbols == (*"hello", "</w>")

    def test_pre_tokenize_multiple_words(self) -> None:
        """Test pre_tokenize with multiple words."""
        encoder = BytePairEncoder(end_token="</end>")  # noqa: S106
        result = encoder.pre_tokenize("hello world")

        assert len(result) == 2  # noqa: PLR2004
        assert result[0].symbols == (*"hello", "</end>")
        assert result[1].symbols == (*"world", "</end>")

    def test_merge_symbols_with_rules(self) -> None:
        """Test merge_symbols applies rules correctly."""
        encoder = BytePairEncoder(rules=[Bigram("s", "t"), Bigram("b", "i")])
        words = [Word(symbols=tuple("biggest"))]
        result = encoder.merge_symbols(words)

        assert len(result) == 1
        # "biggest" with rules "st" and "bi" should become ["bi", "g", "g", "e", "st"]
        assert result[0].symbols == ("bi", "g", "g", "e", "st")

    def test_merge_symbols_no_rules(self) -> None:
        """Test merge_symbols with no rules returns words unchanged."""
        encoder = BytePairEncoder(rules=[])
        words = [Word(symbols=("a", "b", "c"))]
        result = encoder.merge_symbols(words)

        assert len(result) == 1
        assert result[0].symbols == ("a", "b", "c")

    def test_merge_symbols_multiple_words(self) -> None:
        """Test merge_symbols with multiple words."""
        encoder = BytePairEncoder(rules=[Bigram("a", "b")])
        words = [
            Word(symbols=("a", "b", "c")),
            Word(symbols=("x", "y", "z")),
        ]
        result = encoder.merge_symbols(words)

        assert len(result) == 2  # noqa: PLR2004
        assert result[0].symbols == ("ab", "c")
        assert result[1].symbols == ("x", "y", "z")  # No match

    def test_train_step_adds_rule(self) -> None:
        """Test _train_step adds a new rule and updates vocabulary."""
        encoder = BytePairEncoder()
        word1 = Word(symbols=("a", "b", "c"))
        word2 = Word(symbols=("a", "b", "d"))
        corpus = [word1, word2]

        initial_rules_count = len(encoder.rules)
        initial_vocab_count = len(encoder.vocab)

        result = encoder._train_step(corpus)  # noqa: SLF001

        assert len(encoder.rules) == initial_rules_count + 1
        assert len(encoder.vocab) == initial_vocab_count + 1
        assert len(result) == len(corpus)  # Same number of words returned
        assert "ab" in encoder.vocab
        assert Bigram("a", "b") in encoder.rules

    def test_encode_simple(self) -> None:
        """Test encode with simple text and vocab."""
        encoder = BytePairEncoder(
            vocab=["h", "e", "l", "o", "</w>"],
            rules=[],
            end_token="</w>",  # noqa: S106
            unknown_token="<unk>",  # noqa: S106
        )
        result = encoder.encode("hello")

        assert len(result) == 1
        # All characters in vocab, so no replacements needed
        assert result[0] == "hello</w>"

    def test_encode_multiple_words(self) -> None:
        """Test encode with multiple words."""
        encoder = BytePairEncoder(
            vocab=["h", "e", "l", "o", "w", "r", "d", "</w>"],
            rules=[],
            end_token="</w>",  # noqa: S106
            unknown_token="<unk>",  # noqa: S106
        )
        result = encoder.encode("hello world")

        assert len(result) == 2  # noqa: PLR2004
        assert result[0] == "hello</w>"
        assert result[1] == "world</w>"

    def test_encode_with_unknown_symbols(self) -> None:
        """Test encode replaces unknown symbols."""
        encoder = BytePairEncoder(
            vocab=["h", "e", "l", "</w>"],  # 'o' is not in vocab
            rules=[],
            end_token="</w>",  # noqa: S106
            unknown_token="<unk>",  # noqa: S106
        )
        result = encoder.encode("hello")

        assert len(result) == 1
        # 'o' should be replaced with <unk>
        assert result[0] == "hell<unk></w>"

    def test_encode_with_merge_rules(self) -> None:
        """Test encode applies merge rules."""
        encoder = BytePairEncoder(
            vocab=["t", "e", "</w>", "st"],
            rules=[Bigram("s", "t")],
            end_token="</w>",  # noqa: S106
            unknown_token="<unk>",  # noqa: S106
        )
        result = encoder.encode("test")

        assert len(result) == 1
        # "test" with rule "st" should become "t" + "e" + "st" + "</w>" = "test</w>"
        assert result[0] == "test</w>"

    def test_train_multiple_steps(self) -> None:
        """Test train method learns merge rules progressively from corpus."""
        # Build corpus with word frequencies:
        # hug x10, pug x5, pun x12, bun x4, hugs x5
        corpus = ["hug"] * 10 + ["pug"] * 5 + ["pun"] * 12 + ["bun"] * 4 + ["hugs"] * 5

        encoder = BytePairEncoder()

        # Step 1: Learn "ug" - appears 20 times (hug*10 + pug*5 + hugs*5)
        encoder.train(corpus, max_vocab=9)
        assert len(encoder.vocab) == 9  # noqa: PLR2004
        assert "ug" in encoder.vocab
        assert Bigram("u", "g") in encoder.rules

        # Step 2: Learn "un" - appears 16 times (pun*12 + bun*4)
        encoder.train(corpus, max_vocab=10)
        assert len(encoder.vocab) == 10  # noqa: PLR2004
        assert "un" in encoder.vocab
        assert Bigram("u", "n") in encoder.rules

        # Step 3: Learn "un</w>" - appears 16 times (pun*12 + bun*4)
        # Beats "hug" at 15 times (hug*10 + hugs*5)
        encoder.train(corpus, max_vocab=11)
        assert len(encoder.vocab) == 11  # noqa: PLR2004
        assert "un</w>" in encoder.vocab
        assert Bigram("un", "</w>") in encoder.rules

        # Step 4: Learn "hug" - appears 15 times (hug*10 + hugs*5)
        encoder.train(corpus, max_vocab=12)
        assert len(encoder.vocab) == 12  # noqa: PLR2004
        assert "hug" in encoder.vocab
        assert Bigram("h", "ug") in encoder.rules

        assert encoder.rules == [
            Bigram("u", "g"),
            Bigram("u", "n"),
            Bigram("un", "</w>"),
            Bigram("h", "ug"),
        ]


class TestBytePairEncoderJSONRepository:
    """Tests for BytePairEncoderJSONRepository."""

    def test_save_and_load_e2e(self, tmp_path: Path) -> None:
        """Test end-to-end save and load functionality."""
        # Create and train encoder
        encoder = BytePairEncoder(
            end_token="</w>",  # noqa: S106
            unknown_token="<unk>",  # noqa: S106
        )
        corpus = ["hug"] * 10 + ["pug"] * 5
        encoder.train(corpus, max_vocab=9)

        # Save
        save_path = tmp_path / "encoder.json"
        BytePairEncoderJSONRepository.save(encoder, save_path)

        # Load
        loaded = BytePairEncoderJSONRepository.load(save_path)

        # Verify properties match
        assert loaded.vocab == encoder.vocab
        assert loaded.rules == encoder.rules
        assert loaded.end_token == encoder.end_token
        assert loaded.unknown_token == encoder.unknown_token

        # Verify functional equivalence
        test_text = "hug pug"
        assert loaded.encode(test_text) == encoder.encode(test_text)
