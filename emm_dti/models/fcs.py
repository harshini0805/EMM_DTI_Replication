"""
FCS (Frequent Continuous Subsequence) mining module.

Implements efficient subsequence pattern discovery for SMILES and protein sequences.
"""

from collections import defaultdict
from typing import Set, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class FCSModule:
    """
    Frequent Continuous Subsequence mining using Apriori algorithm.

    Discovers recurring k-length patterns in sequences with support threshold.
    """

    def __init__(self, min_support: float = 0.3):
        """
        Initialize FCS module.

        Args:
            min_support: Minimum support threshold (0.0 to 1.0)
                        Patterns appearing in <min_support fraction of sequences
                        are filtered out.

        Raises:
            ValueError: If min_support is not in valid range
        """
        if not 0.0 <= min_support <= 1.0:
            raise ValueError(f"min_support must be in [0, 1], got {min_support}")

        self.min_support = min_support
        self.frequent_patterns: Dict[int, Set[str]] = defaultdict(set)
        self.support_counts: Dict[str, int] = {}

    def extract_kmers(self, sequence: str, k: int) -> Set[str]:
        """
        Extract all k-length substrings from sequence.

        Args:
            sequence: Input sequence (SMILES or protein)
            k: Subsequence length

        Returns:
            Set of k-length subsequences
        """
        kmers = set()
        for i in range(len(sequence) - k + 1):
            kmers.add(sequence[i : i + k])
        return kmers

    def mine(self, sequences: List[str], max_k: int = 5) -> Dict[int, Set[str]]:
        """
        Discover frequent continuous subsequences.

        Uses Apriori-like algorithm to efficiently find patterns appearing
        in at least min_support fraction of sequences.

        Args:
            sequences: List of input sequences (SMILES or proteins)
            max_k: Maximum subsequence length to mine

        Returns:
            Dictionary mapping k (length) to set of frequent k-mers
        """
        if not sequences:
            logger.warning("No sequences provided for FCS mining")
            return {}

        total_sequences = len(sequences)
        min_count = max(1, int(total_sequences * self.min_support))

        logger.info(
            f"Mining FCS patterns from {total_sequences} sequences "
            f"(min_support={self.min_support}, min_count={min_count})"
        )

        frequent = defaultdict(set)
        self.support_counts = {}

        # Start with k=1
        for k in range(1, max_k + 1):
            candidates: Dict[str, int] = defaultdict(int)

            # Extract all k-mers from sequences
            for sequence in sequences:
                kmers = self.extract_kmers(sequence, k)
                for kmer in kmers:
                    candidates[kmer] += 1

            # Filter by support threshold
            frequent_kmers = {
                kmer: count
                for kmer, count in candidates.items()
                if count >= min_count
            }

            if not frequent_kmers:
                logger.info(f"No frequent {k}-mers found. Stopping search.")
                break

            frequent[k] = set(frequent_kmers.keys())
            self.support_counts.update(frequent_kmers)

            logger.debug(f"Found {len(frequent_kmers)} frequent {k}-mers")

        self.frequent_patterns = frequent
        logger.info(
            f"FCS mining complete. Total patterns found: {sum(len(v) for v in frequent.values())}"
        )

        return frequent

    def get_patterns(self, k: int | None = None) -> Set[str] | Dict[int, Set[str]]:
        """
        Get discovered frequent patterns.

        Args:
            k: Pattern length (None for all)

        Returns:
            Set of patterns if k specified, dict otherwise
        """
        if k is None:
            return self.frequent_patterns
        return self.frequent_patterns.get(k, set())

    def get_support(self, pattern: str) -> int:
        """
        Get support count for a specific pattern.

        Args:
            pattern: Query pattern

        Returns:
            Number of sequences containing this pattern
        """
        return self.support_counts.get(pattern, 0)

    def __len__(self) -> int:
        """Total number of frequent patterns discovered."""
        return sum(len(patterns) for patterns in self.frequent_patterns.values())

    def __repr__(self) -> str:
        """String representation."""
        total = len(self)
        by_length = {k: len(v) for k, v in self.frequent_patterns.items()}
        return f"FCSModule(min_support={self.min_support}, total_patterns={total}, by_length={by_length})"


class FragmentVocabulary:
    """
    Vocabulary mapping for frequent subsequences.

    Converts fragment strings to indices for embedding lookup.
    """

    def __init__(self):
        """Initialize fragment vocabulary."""
        self.fragment2idx: Dict[str, int] = {}
        self.idx2fragment: Dict[int, str] = {}
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"

        # Reserve special tokens
        self.fragment2idx[self.pad_token] = 0
        self.fragment2idx[self.unk_token] = 1
        self.idx2fragment[0] = self.pad_token
        self.idx2fragment[1] = self.unk_token

    def build_from_fcs(self, fcs: FCSModule) -> None:
        """
        Build vocabulary from FCS patterns.

        Args:
            fcs: FCSModule instance with mined patterns
        """
        idx = len(self.fragment2idx)

        for patterns in fcs.get_patterns().values():
            for pattern in patterns:
                if pattern not in self.fragment2idx:
                    self.fragment2idx[pattern] = idx
                    self.idx2fragment[idx] = pattern
                    idx += 1

        logger.info(f"Built fragment vocabulary with {len(self.fragment2idx)} tokens")

    def get_index(self, fragment: str) -> int:
        """
        Get index for fragment, returning UNK if not found.

        Args:
            fragment: Query fragment

        Returns:
            Fragment index
        """
        return self.fragment2idx.get(fragment, self.fragment2idx[self.unk_token])

    def get_fragment(self, idx: int) -> str:
        """
        Get fragment string for index.

        Args:
            idx: Query index

        Returns:
            Fragment string
        """
        return self.idx2fragment.get(idx, self.unk_token)

    def __len__(self) -> int:
        """Vocabulary size."""
        return len(self.fragment2idx)

    def __repr__(self) -> str:
        """String representation."""
        return f"FragmentVocabulary(size={len(self)})"
