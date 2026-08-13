"""
Unit tests for FCS (Frequent Continuous Subsequence) mining.

Tests the pattern discovery and vocabulary building pipeline.
"""

import pytest
from emm_dti.models.fcs import FCSModule, FragmentVocabulary


class TestFCSModule:
    """Tests for FCSModule pattern mining."""

    def test_kmers_extraction(self):
        """Test k-mer extraction from sequences."""
        fcs = FCSModule(min_support=0.0)
        sequence = "ABCD"

        # Test 1-mers
        kmers = fcs.extract_kmers(sequence, k=1)
        assert kmers == {"A", "B", "C", "D"}

        # Test 2-mers
        kmers = fcs.extract_kmers(sequence, k=2)
        assert kmers == {"AB", "BC", "CD"}

        # Test 3-mers
        kmers = fcs.extract_kmers(sequence, k=3)
        assert kmers == {"ABC", "BCD"}

    def test_mining_simple(self):
        """Test FCS mining on simple data."""
        fcs = FCSModule(min_support=0.5)  # 50% support threshold

        sequences = [
            "ABABAB",  # Contains AB 3 times
            "ABAB",    # Contains AB 2 times
            "CDCD",    # Contains CD 2 times
        ]

        patterns = fcs.mine(sequences, max_k=2)

        # AB appears in 2/3 = 66% → should be frequent
        assert "AB" in patterns[2]

        # CD appears in 1/3 = 33% → should NOT be frequent
        assert "CD" not in patterns.get(2, set())

    def test_min_support_threshold(self):
        """Test that min_support threshold is respected."""
        sequences = ["AAA", "AAA", "BBB"]  # A appears in 2/3 = 66%

        # High threshold: only very common patterns
        fcs_high = FCSModule(min_support=0.7)
        patterns_high = fcs_high.mine(sequences, max_k=1)
        # "A" appears in 66% < 70% → not frequent
        assert "A" not in patterns_high.get(1, set())

        # Low threshold: more patterns
        fcs_low = FCSModule(min_support=0.5)
        patterns_low = fcs_low.mine(sequences, max_k=1)
        # "A" appears in 66% > 50% → frequent
        assert "A" in patterns_low.get(1, set())

    def test_empty_sequences(self):
        """Test handling of empty input."""
        fcs = FCSModule(min_support=0.3)
        patterns = fcs.mine([], max_k=1)
        assert patterns == {}

    def test_invalid_support(self):
        """Test that invalid support values raise errors."""
        with pytest.raises(ValueError):
            FCSModule(min_support=-0.1)

        with pytest.raises(ValueError):
            FCSModule(min_support=1.5)

    def test_get_support(self):
        """Test pattern support counting."""
        fcs = FCSModule(min_support=0.0)
        sequences = ["AAABBB", "AAACCC", "DDD"]

        fcs.mine(sequences, max_k=2)

        # AA appears in 2/3 sequences
        assert fcs.get_support("AA") == 2

        # DDD doesn't appear (we only mine up to 2-mers)
        assert fcs.get_support("DDD") == 0


class TestFragmentVocabulary:
    """Tests for FragmentVocabulary."""

    def test_initialization(self):
        """Test vocabulary initialization with special tokens."""
        vocab = FragmentVocabulary()

        assert len(vocab) == 2  # <PAD> and <UNK>
        assert vocab.get_index("<PAD>") == 0
        assert vocab.get_index("<UNK>") == 1

    def test_building_from_fcs(self):
        """Test building vocabulary from FCS patterns."""
        fcs = FCSModule(min_support=0.1)
        sequences = ["AAABBB", "BBBCCC"]
        fcs.mine(sequences, max_k=2)

        vocab = FragmentVocabulary()
        vocab.build_from_fcs(fcs)

        # Should have special tokens + all patterns
        assert len(vocab) > 2
        # Patterns should be in vocabulary
        assert vocab.get_index("A") > 1

    def test_index_lookup(self):
        """Test bidirectional index-fragment lookup."""
        vocab = FragmentVocabulary()
        vocab.fragment2idx["TEST"] = 10
        vocab.idx2fragment[10] = "TEST"

        assert vocab.get_index("TEST") == 10
        assert vocab.get_fragment(10) == "TEST"

    def test_unknown_fragment(self):
        """Test handling of unknown fragments."""
        vocab = FragmentVocabulary()

        # Unknown fragment should return UNK index
        unk_idx = vocab.get_index("UNKNOWN_PATTERN")
        assert unk_idx == vocab.fragment2idx["<UNK>"]

        # Unknown index should return UNK token
        unk_token = vocab.get_fragment(9999)
        assert unk_token == "<UNK>"

    def test_vocabulary_size(self):
        """Test vocabulary size tracking."""
        vocab = FragmentVocabulary()
        initial_size = len(vocab)

        # Add some fragments
        vocab.fragment2idx["PATTERN1"] = len(vocab)
        vocab.idx2fragment[len(vocab) - 1] = "PATTERN1"

        assert len(vocab) == initial_size + 1


class TestFCSIntegration:
    """Integration tests for FCS mining pipeline."""

    def test_full_pipeline(self):
        """Test complete FCS mining and vocabulary building."""
        # Simulate real sequences
        smiles_list = [
            "CC(=O)Oc1ccccc1",
            "CN1C(=O)CC(c2ccccc2)C1=O",
            "CC(=O)Oc1ccccc1C(=O)O",
        ]

        # Mine patterns
        fcs = FCSModule(min_support=0.3)
        patterns = fcs.mine(smiles_list, max_k=3)

        # Build vocabulary
        vocab = FragmentVocabulary()
        vocab.build_from_fcs(fcs)

        # Validate
        assert len(patterns) > 0
        assert len(vocab) > 2
        print(f"✓ Mined {len(fcs)} patterns")
        print(f"✓ Built vocabulary with {len(vocab)} tokens")

    def test_greedy_tokenization(self):
        """Test greedy matching tokenization strategy."""
        fcs = FCSModule(min_support=0.0)
        sequences = ["AAABBBCCC", "AAABBB", "BBBCCC"]
        fcs.mine(sequences, max_k=2)

        vocab = FragmentVocabulary()
        vocab.build_from_fcs(fcs)

        # Test greedy matching
        sequence = "AAABBBCCC"
        patterns_by_len = sorted(fcs.get_patterns().keys(), reverse=True)
        patterns_list = []
        for length in patterns_by_len:
            patterns_list.extend(fcs.get_patterns(length))

        # Manually implement greedy tokenization
        tokens = []
        i = 0
        while i < len(sequence):
            matched = False
            for pattern in sorted(patterns_list, key=len, reverse=True):
                if sequence[i:].startswith(pattern):
                    tokens.append(vocab.get_index(pattern))
                    i += len(pattern)
                    matched = True
                    break

            if not matched:
                # Fallback to character
                tokens.append(vocab.get_index(sequence[i]))
                i += 1

        # Should have tokenized the sequence
        assert len(tokens) > 0
        print(f"✓ Tokenized '{sequence}' into {len(tokens)} tokens")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
