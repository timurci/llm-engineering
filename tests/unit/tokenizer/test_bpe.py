"""Tests for Byte Pair Encoder (BPE) implementation."""

from collections import Counter

from llm_engineering.tokenizer.bpe import Bigram, Word


class TestWord:
    """Tests for the Word dataclass."""

    def test_from_str(self) -> None:
        """Test creating Word from string with various words and end tokens."""
        assert Word.from_str("apple", "</end>").symbols == [
            "a",
            "p",
            "p",
            "l",
            "e",
            "</end>",
        ]
        assert Word.from_str("a", "</w>").symbols == ["a", "</w>"]
        assert Word.from_str("", "</w>").symbols == ["</w>"]
        assert Word.from_str("hello", "<END>").symbols == [
            "h",
            "e",
            "l",
            "l",
            "o",
            "<END>",
        ]
        assert Word.from_str("world", "$").symbols == ["w", "o", "r", "l", "d", "$"]

    def test_count_pairs(self) -> None:
        """Test counting adjacent pairs of symbols with various scenarios."""
        # Single pair
        word = Word(symbols=["un", "help", "ful"])
        pairs = word.count_pairs()
        assert pairs == Counter(
            {
                Bigram("un", "help"): 1,
                Bigram("help", "ful"): 1,
            }
        )

        # Multiple of the same adjacent pairs
        word2 = Word(symbols=["a", "b", "a", "b", "a", "b"])
        pairs2 = word2.count_pairs()
        assert pairs2 == Counter(
            {
                Bigram("a", "b"): 3,
                Bigram("b", "a"): 2,
            }
        )

        # Word with all same adjacent pairs
        word3 = Word(symbols=["x", "x", "x", "x"])
        pairs3 = word3.count_pairs()
        assert pairs3 == Counter({Bigram("x", "x"): 3})

        # Single symbol (no pairs)
        word4 = Word(symbols=["a"])
        pairs4 = word4.count_pairs()
        assert pairs4 == Counter()

        # Empty word
        word5 = Word(symbols=[])
        pairs5 = word5.count_pairs()
        assert pairs5 == Counter()

    def test_replace_missing_symbols(self) -> None:
        """Test replacing unknown symbols with missing token."""
        word = Word(symbols=["a", "b", "c", "d", "e"])
        vocab = ["a", "c", "e"]
        missing_token = "<unk>"  # noqa: S105

        result = word.replace_missing_symbols(vocab, missing_token)
        assert result.symbols == ["a", "<unk>", "c", "<unk>", "e"]

        # Test with empty vocab (all symbols replaced)
        result2 = word.replace_missing_symbols([], missing_token)
        assert result2.symbols == ["<unk>", "<unk>", "<unk>", "<unk>", "<unk>"]

        # Test with all symbols in vocab (no replacements)
        result3 = word.replace_missing_symbols(["a", "b", "c", "d", "e"], missing_token)
        assert result3.symbols == ["a", "b", "c", "d", "e"]

    def test_merge_pairs(self) -> None:
        """Test merging adjacent pairs according to rules with various scenarios."""
        word = Word(symbols=list("biggest"))

        ruleset1 = [Bigram("s", "t")]
        ruleset2 = [*ruleset1, Bigram("b", "i")]
        ruleset3 = [*ruleset2, Bigram("bi", "g")]

        assert word.merge_pairs(ruleset1).symbols == [*list("bigge"), "st"]
        assert word.merge_pairs(ruleset2).symbols == ["bi", *list("gge"), "st"]
        assert word.merge_pairs(ruleset3).symbols == ["big", "g", "e", "st"]

        # Multiple occurrences of same bigram
        word2 = Word(symbols=["a", "b", "a", "b", "a", "b"])
        ruleset4 = [Bigram("a", "b")]
        result = word2.merge_pairs(ruleset4)
        assert result.symbols == ["ab", "ab", "ab"]

        # No matching rules
        word3 = Word(symbols=["a", "b", "c"])
        ruleset5 = [Bigram("x", "y")]
        result2 = word3.merge_pairs(ruleset5)
        assert result2.symbols == ["a", "b", "c"]

        # Empty ruleset
        result3 = word3.merge_pairs([])
        assert result3.symbols == ["a", "b", "c"]
