#!/usr/bin/env python3
"""
Dataset Organization Script for EMM-DTI Replication.

Organizes raw pre-split benchmark datasets (human_random, biosnap_random, celegans_random, bindingdb_random)
into standardized directory structure in data/:
  data/<dataset_name>/
    ├── drugs.csv (drug_id, smiles)
    ├── proteins.csv (protein_id, sequence)
    └── interactions.csv (drug_id, protein_id, interaction, split)
"""

import os
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent


def organize_presplit_dataset(source_folder: str, target_name: str) -> bool:
    """
    Organize a pre-split dataset (containing train/, valid/, test/ subdirectories)
    into data/<target_name>/ with drugs.csv, proteins.csv, and interactions.csv (with split column).
    """
    logger.info("=" * 80)
    logger.info(f"Organizing dataset: {source_folder} -> data/{target_name}")
    logger.info("=" * 80)

    source_dir = PROJECT_ROOT / source_folder
    output_dir = PROJECT_ROOT / "data" / target_name

    if not source_dir.exists():
        logger.warning(f"Source directory not found: {source_dir}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_to_id = {}
    seq_to_id = {}
    drugs_list = []
    proteins_list = []
    interactions_list = []

    splits = ["train", "valid", "test"]

    for split in splits:
        split_dir = source_dir / split
        if not split_dir.exists():
            if split == "valid" and (source_dir / "val").exists():
                split_dir = source_dir / "val"
            else:
                logger.warning(f"  Missing split subdirectory: {split_dir}")
                continue

        smiles_file = split_dir / "smiles.csv"
        seq_file = split_dir / "sequence.csv"
        samples_file = split_dir / "samples.csv"

        if not (smiles_file.exists() and seq_file.exists() and samples_file.exists()):
            logger.warning(f"  Missing required files in {split_dir}")
            continue

        try:
            smiles_df = pd.read_csv(smiles_file)
            seq_df = pd.read_csv(seq_file)
            samples_df = pd.read_csv(samples_file)

            smiles_map = dict(zip(smiles_df['index'], smiles_df['smiles']))
            seq_map = dict(zip(seq_df['index'], seq_df['sequence']))

            logger.info(f"  Processing {split} split: {len(samples_df):,} samples...")

            for _, row in samples_df.iterrows():
                smiles_idx = int(row['smiles'])
                seq_idx = int(row['sequence'])
                label = int(row['interactions'])

                smiles_str = str(smiles_map[smiles_idx]).strip()
                seq_str = str(seq_map[seq_idx]).strip()

                if smiles_str not in smiles_to_id:
                    drug_id = f"D{len(smiles_to_id) + 1:05d}"
                    smiles_to_id[smiles_str] = drug_id
                    drugs_list.append({"drug_id": drug_id, "smiles": smiles_str})

                if seq_str not in seq_to_id:
                    protein_id = f"P{len(seq_to_id) + 1:05d}"
                    seq_to_id[seq_str] = protein_id
                    proteins_list.append({"protein_id": protein_id, "sequence": seq_str})

                interactions_list.append({
                    "drug_id": smiles_to_id[smiles_str],
                    "protein_id": seq_to_id[seq_str],
                    "interaction": label,
                    "split": split,
                })

        except Exception as e:
            logger.error(f"  Error processing split {split} in {source_folder}: {e}")
            return False

    if not interactions_list:
        logger.error(f"No interactions found for {source_folder}")
        return False

    # Save to CSV files
    drugs_df_out = pd.DataFrame(drugs_list)
    proteins_df_out = pd.DataFrame(proteins_list)
    interactions_df_out = pd.DataFrame(interactions_list)

    drugs_df_out.to_csv(output_dir / "drugs.csv", index=False)
    proteins_df_out.to_csv(output_dir / "proteins.csv", index=False)
    interactions_df_out.to_csv(output_dir / "interactions.csv", index=False)

    total_samples = len(interactions_df_out)
    split_counts = interactions_df_out["split"].value_counts().to_dict()
    split_ratios = {k: f"{v / total_samples * 100:.1f}% ({v:,})" for k, v in split_counts.items()}

    logger.info(f"✓ Successfully organized {target_name}:")
    logger.info(f"   - Unique Drugs: {len(drugs_df_out):,}")
    logger.info(f"   - Unique Proteins: {len(proteins_df_out):,}")
    logger.info(f"   - Total Interactions: {total_samples:,}")
    logger.info(f"   - Split Breakdown: {split_ratios}\n")

    return True


def main():
    """Organize all available benchmark datasets."""
    dataset_mappings = [
        ("human_random", "human"),
        ("biosnap_random", "biosnap"),
        ("celegans_random", "celegans"),
        ("bindingdb_random", "bindingdb"),
    ]

    results = {}
    for source, target in dataset_mappings:
        if (PROJECT_ROOT / source).exists():
            results[target] = organize_presplit_dataset(source, target)
        else:
            logger.info(f"Skipping {source} (folder not present)")

    logger.info("=" * 80)
    logger.info("Dataset Organization Summary")
    logger.info("=" * 80)
    for target, success in results.items():
        status = "✓ Ready" if success else "✗ Failed"
        logger.info(f"  {target:<15}: {status}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
