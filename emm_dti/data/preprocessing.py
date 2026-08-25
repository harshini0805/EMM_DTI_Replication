"""
Data preprocessing utilities for DTI prediction.

Handles SMILES canonicalization, protein sequence cleaning, and data validation.
"""

import logging
from typing import Tuple, Optional

try:
    from rdkit import Chem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False
    logging.warning("RDKit not available. SMILES canonicalization disabled.")

logger = logging.getLogger(__name__)


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """
    Canonicalize SMILES string using RDKit.

    Canonical SMILES provides a unique representation for the same molecule,
    ensuring consistency across datasets.

    Args:
        smiles: SMILES string

    Returns:
        Canonical SMILES or None if invalid

    Raises:
        ValueError: If RDKit is not installed
    """
    if not HAS_RDKIT:
        raise ValueError(
            "RDKit required for SMILES canonicalization. "
            "Install with: pip install rdkit"
        )

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logger.warning(f"Invalid SMILES: {smiles}")
            return None
        return Chem.MolToSmiles(mol)
    except Exception as e:
        logger.warning(f"Error canonicalizing SMILES '{smiles}': {e}")
        return None


def clean_protein_sequence(sequence: str) -> str:
    """
    Clean protein sequence.

    Removes whitespace and converts to uppercase. Validates that sequence
    contains only standard amino acids.

    Args:
        sequence: Protein sequence

    Returns:
        Cleaned sequence

    Raises:
        ValueError: If sequence contains invalid characters
    """
    sequence = sequence.strip().upper()

    # Standard amino acids
    valid_aa = set("ACDEFGHIKLMNPQRSTVWY")

    invalid_chars = set(sequence) - valid_aa
    if invalid_chars:
        raise ValueError(
            f"Invalid amino acids in sequence: {invalid_chars}. "
            f"Expected only standard 20 amino acids."
        )

    return sequence


def validate_interaction(
    drug_id: str,
    protein_id: str,
    interaction: int,
) -> Tuple[str, str, int]:
    """
    Validate interaction data.

    Args:
        drug_id: Drug identifier
        protein_id: Protein identifier
        interaction: Binary interaction label (0 or 1)

    Returns:
        Validated tuple

    Raises:
        ValueError: If data is invalid
    """
    if not drug_id or not protein_id:
        raise ValueError("Drug ID and Protein ID cannot be empty")

    if interaction not in [0, 1]:
        raise ValueError(f"Interaction label must be 0 or 1, got {interaction}")

    return drug_id, protein_id, interaction


class DataPreprocessor:
    """
    Batch preprocessing for DTI datasets.

    Handles SMILES canonicalization and sequence validation at scale.
    """

    def __init__(self, canonicalize_smiles: bool = True):
        """
        Initialize preprocessor.

        Args:
            canonicalize_smiles: Whether to canonicalize SMILES strings
        """
        self.canonicalize = canonicalize_smiles
        self.invalid_count = 0
        self.valid_count = 0

    def process_smiles(self, smiles_list: list[str]) -> list[Optional[str]]:
        """
        Process list of SMILES strings.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            List of processed SMILES (None for invalid entries)
        """
        processed = []

        for smiles in smiles_list:
            if self.canonicalize:
                canonical = canonicalize_smiles(smiles)
                if canonical is None:
                    self.invalid_count += 1
                    processed.append(None)
                else:
                    self.valid_count += 1
                    processed.append(canonical)
            else:
                processed.append(smiles)

        return processed

    def process_sequences(self, sequences: list[str]) -> list[str]:
        """
        Process list of protein sequences.

        Args:
            sequences: List of protein sequences

        Returns:
            List of cleaned sequences
        """
        processed = []

        for seq in sequences:
            try:
                cleaned = clean_protein_sequence(seq)
                self.valid_count += 1
                processed.append(cleaned)
            except ValueError as e:
                logger.warning(f"Error processing sequence: {e}")
                self.invalid_count += 1
                processed.append(None)

        return processed

    def get_stats(self) -> dict:
        """
        Get preprocessing statistics.

        Returns:
            Dictionary with stats
        """
        total = self.valid_count + self.invalid_count
        return {
            "total": total,
            "valid": self.valid_count,
            "invalid": self.invalid_count,
            "valid_ratio": self.valid_count / max(total, 1),
        }

    def reset_stats(self) -> None:
        """Reset counters."""
        self.valid_count = 0
        self.invalid_count = 0
