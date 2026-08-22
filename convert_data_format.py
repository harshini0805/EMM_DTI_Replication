"""
Convert data from train/test/validation folder format to DTI format.

Input structure:
  train/
    samples.csv (smiles, sequence, interactions)
    sequences.csv (index, sequence)
    smiles.csv (index, smiles)
  test/
    ...
  validation/
    ...

Output structure:
  data/human/
    drugs.csv (drug_id, smiles)
    proteins.csv (protein_id, sequence)
    interactions.csv (drug_id, protein_id, interaction)
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


def convert_data():
    """Convert data from existing format to DTI format."""

    project_root = Path(__file__).parent

    # Find data directory (could be train/test/val or human_random)
    human_random_dir = project_root / "human_random"

    # Output path
    output_dir = project_root / "data" / "human"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Converting Data Format")
    logger.info("=" * 80)

    # Check if human_random exists
    if human_random_dir.exists():
        logger.info(f"Found human_random directory: {human_random_dir}")

        # List subdirectories
        subdirs = [d for d in human_random_dir.iterdir() if d.is_dir()]
        logger.info(f"Subdirectories: {[d.name for d in subdirs]}")

        # Use subdirectories as splits
        split_dirs = {d.name: d for d in subdirs}
    else:
        # Fall back to train/test/validation
        train_dir = project_root / "train"
        test_dir = project_root / "test"
        val_dir = project_root / "validation"

        for split_dir in [train_dir, test_dir, val_dir]:
            if not split_dir.exists():
                logger.error(f"Missing directory: {split_dir}")
                logger.error(f"Expected either: {human_random_dir} or separate train/test/validation folders")
                return False

        split_dirs = {"train": train_dir, "test": test_dir, "validation": val_dir}

    logger.info(f"\nInput directories found:")
    for name, path in split_dirs.items():
        logger.info(f"  - {name}: {path}")

    # Load all data
    logger.info("\nLoading data...")

    all_drugs = {}  # smiles_idx -> smiles
    all_proteins = {}  # seq_idx -> sequence
    all_interactions = []  # (smiles_idx, seq_idx, interaction)

    for split_name, split_dir in split_dirs.items():
        logger.info(f"\n  Processing {split_name}...")

        try:
            # Load samples (interactions)
            samples_path = split_dir / "samples.csv"
            samples_df = pd.read_csv(samples_path)
            logger.info(f"    Loaded {len(samples_df)} samples")

            # Load SMILES mapping
            smiles_path = split_dir / "smiles.csv"
            smiles_df = pd.read_csv(smiles_path)
            logger.info(f"    Loaded {len(smiles_df)} SMILES")

            # Load sequences mapping
            sequences_path = split_dir / "sequence.csv"
            sequences_df = pd.read_csv(sequences_path)
            logger.info(f"    Loaded {len(sequences_df)} sequences")

            # Process samples
            for idx, row in samples_df.iterrows():
                smiles_idx = int(row['smiles'])
                seq_idx = int(row['sequence'])
                interaction = int(row['interactions'])

                # Get actual SMILES and sequence
                if smiles_idx not in all_drugs:
                    smiles = smiles_df[smiles_df['index'] == smiles_idx]['smiles'].values[0]
                    all_drugs[smiles_idx] = smiles

                if seq_idx not in all_proteins:
                    sequence = sequences_df[sequences_df['index'] == seq_idx]['sequence'].values[0]
                    all_proteins[seq_idx] = sequence

                all_interactions.append((smiles_idx, seq_idx, interaction, split_name))

        except Exception as e:
            logger.error(f"  Error processing {split_name}: {e}")
            return False

    logger.info(f"\nTotal data collected:")
    logger.info(f"  - Unique drugs: {len(all_drugs)}")
    logger.info(f"  - Unique proteins: {len(all_proteins)}")
    logger.info(f"  - Total interactions: {len(all_interactions)}")

    # Create output DataFrames
    logger.info("\nCreating output format...")

    # Drugs CSV
    drugs_data = []
    smiles_id_map = {}  # old_idx -> new_drug_id
    for new_id, (old_idx, smiles) in enumerate(all_drugs.items()):
        smiles_id_map[old_idx] = f"D{new_id:05d}"
        drugs_data.append({"drug_id": f"D{new_id:05d}", "smiles": smiles})

    drugs_df = pd.DataFrame(drugs_data)
    drugs_path = output_dir / "drugs.csv"
    drugs_df.to_csv(drugs_path, index=False)
    logger.info(f"✓ Saved {len(drugs_df)} drugs to {drugs_path}")

    # Proteins CSV
    proteins_data = []
    seq_id_map = {}  # old_idx -> new_protein_id
    for new_id, (old_idx, sequence) in enumerate(all_proteins.items()):
        seq_id_map[old_idx] = f"P{new_id:05d}"
        proteins_data.append({"protein_id": f"P{new_id:05d}", "sequence": sequence})

    proteins_df = pd.DataFrame(proteins_data)
    proteins_path = output_dir / "proteins.csv"
    proteins_df.to_csv(proteins_path, index=False)
    logger.info(f"✓ Saved {len(proteins_df)} proteins to {proteins_path}")

    # Interactions CSV
    interactions_data = []
    for smiles_idx, seq_idx, interaction, split_name in all_interactions:
        drug_id = smiles_id_map[smiles_idx]
        protein_id = seq_id_map[seq_idx]
        mapped_split = "val" if split_name in ["validation", "valid"] else split_name
        interactions_data.append({
            "drug_id": drug_id,
            "protein_id": protein_id,
            "interaction": interaction,
            "split": mapped_split
        })

    interactions_df = pd.DataFrame(interactions_data)
    interactions_path = output_dir / "interactions.csv"
    interactions_df.to_csv(interactions_path, index=False)
    logger.info(f"✓ Saved {len(interactions_df)} interactions to {interactions_path}")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Conversion Complete!")
    logger.info("=" * 80)
    logger.info(f"\nOutput files created in: {output_dir}")
    logger.info(f"  - drugs.csv ({len(drugs_df)} unique drugs)")
    logger.info(f"  - proteins.csv ({len(proteins_df)} unique proteins)")
    logger.info(f"  - interactions.csv ({len(interactions_df)} interactions)")

    # Data statistics
    positive = (interactions_df['interaction'] == 1).sum()
    negative = (interactions_df['interaction'] == 0).sum()
    logger.info(f"\nInteraction statistics:")
    logger.info(f"  - Positive: {positive} ({positive/len(interactions_df)*100:.1f}%)")
    logger.info(f"  - Negative: {negative} ({negative/len(interactions_df)*100:.1f}%)")

    logger.info("\n" + "=" * 80)
    logger.info("Ready to train! Run:")
    logger.info("  python -m emm_dti.train --config configs/train_human.yaml")
    logger.info("=" * 80 + "\n")

    return True


if __name__ == "__main__":
    success = convert_data()
    exit(0 if success else 1)
