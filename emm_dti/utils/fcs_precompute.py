"""
Utility script to pre-compute FCS patterns.

Optional: For reproducibility and faster subsequent runs,
pre-compute FCS patterns and cache them.
"""

import argparse
from pathlib import Path
import pickle
import logging

from emm_dti.utils.logging_utils import setup_logging
from emm_dti.models.fcs import FCSModule, FragmentVocabulary
import pandas as pd


logger = logging.getLogger(__name__)


def precompute_fcs(
    data_dir: str | Path,
    min_support: float = 0.3,
    max_k: int = 3,
    output_path: str | Path | None = None,
) -> None:
    """
    Pre-compute FCS patterns for a dataset and cache them.

    Args:
        data_dir: Directory containing drugs.csv, proteins.csv, interactions.csv
        min_support: Minimum support threshold (0.0 to 1.0)
        max_k: Maximum k-mer length to mine
        output_path: Path to save cached patterns (default: data_dir/.fcs_cache.pkl)
    """
    data_dir = Path(data_dir)
    output_path = output_path or data_dir / ".fcs_cache.pkl"

    logger.info("=" * 80)
    logger.info("FCS Pattern Pre-computation")
    logger.info("=" * 80)

    # Load sequences
    logger.info(f"Loading sequences from {data_dir}")

    try:
        drugs_df = pd.read_csv(data_dir / "drugs.csv")
        proteins_df = pd.read_csv(data_dir / "proteins.csv")

        drug_sequences = drugs_df["smiles"].tolist()
        protein_sequences = proteins_df["sequence"].tolist()

        all_sequences = drug_sequences + protein_sequences
        logger.info(f"Loaded {len(drug_sequences)} drugs + {len(protein_sequences)} proteins")

    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        return

    # Mine patterns
    logger.info(f"\nMining FCS patterns (min_support={min_support}, max_k={max_k})...")

    fcs = FCSModule(min_support=min_support)
    patterns = fcs.mine(all_sequences, max_k=max_k)

    logger.info("Mining complete:")
    total_patterns = 0
    for k in sorted(patterns.keys()):
        count = len(patterns[k])
        total_patterns += count
        logger.info(f"  {k}-mers: {count} patterns")

    logger.info(f"Total patterns mined: {total_patterns}")

    # Build vocabulary
    logger.info("\nBuilding vocabulary...")

    vocab = FragmentVocabulary()
    vocab.build_from_fcs(fcs)

    logger.info(f"Vocabulary size: {len(vocab)} tokens")

    # Cache
    logger.info(f"\nCaching to {output_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "wb") as f:
        pickle.dump(fcs, f)

    logger.info("✓ FCS patterns cached successfully")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Summary:")
    logger.info(f"  Patterns mined: {total_patterns}")
    logger.info(f"  Vocabulary size: {len(vocab)}")
    logger.info(f"  Cache location: {output_path}")
    logger.info("=" * 80)

    logger.info("\nNext time you run training, cached patterns will be loaded automatically:")
    logger.info(f"  python -m emm_dti.train --data_dir {data_dir}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Pre-compute and cache FCS patterns for a dataset"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/human",
        help="Directory containing data files",
    )

    parser.add_argument(
        "--min_support",
        type=float,
        default=0.3,
        help="Minimum support threshold (0.0 to 1.0)",
    )

    parser.add_argument(
        "--max_k",
        type=int,
        default=3,
        help="Maximum k-mer length to mine",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output cache file path (default: data_dir/.fcs_cache.pkl)",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    # Pre-compute
    precompute_fcs(
        data_dir=args.data_dir,
        min_support=args.min_support,
        max_k=args.max_k,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
