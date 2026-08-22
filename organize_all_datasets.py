#!/usr/bin/env python3
"""
Complete Dataset Organization for EMM-DTI

Organizes multiple datasets into standard format:
  data/<dataset_name>/
    ├── drugs.csv (drug_id, smiles)
    ├── proteins.csv (protein_id, sequence)
    └── interactions.csv (drug_id, protein_id, interaction)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import sys

def organize_random_split(source_folder, output_name):
    """
    Organize datasets with train/test/valid split structure.

    Expected structure:
    source_folder/
    ├── train/
    │   ├── smiles.csv
    │   ├── sequence.csv
    │   └── samples.csv
    ├── test/
    └── valid/
    """
    print(f"\n{'='*80}")
    print(f"Organizing {output_name} dataset...")
    print(f"{'='*80}")

    root = Path("D:\\Projects\\EMM_DTI_Replication")
    source_dir = root / source_folder
    output_dir = root / "data" / output_name

    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load all splits
        splits = {}
        for split in ["train", "test", "valid"]:
            split_dir = source_dir / split
            if split_dir.exists():
                smiles = pd.read_csv(split_dir / "smiles.csv", header=None)
                sequence = pd.read_csv(split_dir / "sequence.csv", header=None)
                samples = pd.read_csv(split_dir / "samples.csv", header=None)
                splits[split] = {"smiles": smiles, "sequence": sequence, "samples": samples}
                print(f"  Loaded {split}: {len(smiles)} samples")
            else:
                print(f"  ⚠️  Missing {split} split")

        if not splits:
            print(f"❌ No valid splits found in {source_dir}")
            return False

        # Combine all splits but track their original splits
        all_smiles_list = []
        all_sequence_list = []
        all_samples_list = []
        split_labels = []

        for split_name, s in splits.items():
            all_smiles_list.append(s["smiles"])
            all_sequence_list.append(s["sequence"])
            all_samples_list.append(s["samples"])
            
            # Map 'valid' to 'val' for our loader
            mapped_split = 'val' if split_name == 'valid' else split_name
            split_labels.extend([mapped_split] * len(s["samples"]))

        all_smiles = pd.concat(all_smiles_list, ignore_index=True)
        all_sequence = pd.concat(all_sequence_list, ignore_index=True)
        all_samples = pd.concat(all_samples_list, ignore_index=True)

        print(f"  Total combined: {len(all_smiles)} samples")

        # Create drugs.csv
        drugs_df = pd.DataFrame({
            'drug_id': [f'D{i:05d}' for i in range(len(all_smiles))],
            'smiles': all_smiles.iloc[:, 0].values
        })
        drugs_df.to_csv(output_dir / "drugs.csv", index=False)
        print(f"  ✅ Created drugs.csv ({len(drugs_df)} unique drugs)")

        # Create proteins.csv
        proteins_df = pd.DataFrame({
            'protein_id': [f'P{i:05d}' for i in range(len(all_sequence))],
            'sequence': all_sequence.iloc[:, 0].values
        })
        proteins_df.to_csv(output_dir / "proteins.csv", index=False)
        print(f"  ✅ Created proteins.csv ({len(proteins_df)} unique proteins)")

        # Create interactions.csv with split column preserved!
        interactions_df = pd.DataFrame({
            'drug_id': [f'D{i:05d}' for i in range(len(all_samples))],
            'protein_id': [f'P{i:05d}' for i in range(len(all_samples))],
            'interaction': all_samples.iloc[:, 0].values,
            'split': split_labels
        })
        interactions_df.to_csv(output_dir / "interactions.csv", index=False)

        pos_count = (interactions_df['interaction'] == 1).sum()
        neg_count = (interactions_df['interaction'] == 0).sum()
        print(f"  ✅ Created interactions.csv ({len(interactions_df)} total)")
        print(f"     - Positive: {pos_count}")
        print(f"     - Negative: {neg_count}")

        return True

    except Exception as e:
        print(f"❌ Error organizing {output_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def organize_drugbank_or_enzyme(source_folder, output_name):
    """
    Organize single-file datasets (drugbank, enzyme).

    Expected structure:
    source_folder/
    └── Drug_Target_Pair_*.csv
    """
    print(f"\n{'='*80}")
    print(f"Organizing {output_name} dataset...")
    print(f"{'='*80}")

    root = Path("D:\\Projects\\EMM_DTI_Replication")
    source_dir = root / source_folder
    output_dir = root / "data" / output_name

    if not source_dir.exists():
        print(f"❌ Source directory not found: {source_dir}")
        return False

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Find the main CSV file
        csv_files = list(source_dir.glob("*.csv"))
        if not csv_files:
            print(f"❌ No CSV files found in {source_dir}")
            return False

        df = pd.read_csv(csv_files[0])
        print(f"  Loaded: {csv_files[0].name}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")
        print(f"  First row:\n{df.iloc[0]}")

        # Try to auto-detect columns
        col_names = [c.lower() for c in df.columns]

        # Find SMILES column
        smiles_col = None
        for col in df.columns:
            if 'smiles' in col.lower() or 'smile' in col.lower():
                smiles_col = col
                break

        # Find sequence column
        seq_col = None
        for col in df.columns:
            if 'sequence' in col.lower() or 'seq' in col.lower() or 'protein' in col.lower():
                seq_col = col
                break

        # Find interaction column
        inter_col = None
        for col in df.columns:
            if 'interaction' in col.lower() or 'label' in col.lower() or 'active' in col.lower():
                inter_col = col
                break

        if not all([smiles_col, seq_col, inter_col]):
            print(f"\n⚠️  Could not auto-detect all columns!")
            print(f"  SMILES column: {smiles_col}")
            print(f"  Sequence column: {seq_col}")
            print(f"  Interaction column: {inter_col}")
            print(f"\n  Please manually edit this script to specify columns.")
            print(f"  Available columns: {list(df.columns)}")
            return False

        # Create drugs.csv
        drugs_df = pd.DataFrame({
            'drug_id': [f'D{i:05d}' for i in range(len(df))],
            'smiles': df[smiles_col].values
        })
        drugs_df.to_csv(output_dir / "drugs.csv", index=False)
        print(f"  ✅ Created drugs.csv ({len(drugs_df)} drugs)")

        # Create proteins.csv
        proteins_df = pd.DataFrame({
            'protein_id': [f'P{i:05d}' for i in range(len(df))],
            'sequence': df[seq_col].values
        })
        proteins_df.to_csv(output_dir / "proteins.csv", index=False)
        print(f"  ✅ Created proteins.csv ({len(proteins_df)} proteins)")

        # Create interactions.csv
        interactions_df = pd.DataFrame({
            'drug_id': [f'D{i:05d}' for i in range(len(df))],
            'protein_id': [f'P{i:05d}' for i in range(len(df))],
            'interaction': df[inter_col].values
        })
        interactions_df.to_csv(output_dir / "interactions.csv", index=False)

        pos_count = (interactions_df['interaction'] == 1).sum()
        neg_count = (interactions_df['interaction'] == 0).sum()
        print(f"  ✅ Created interactions.csv ({len(interactions_df)} total)")
        print(f"     - Positive: {pos_count}")
        print(f"     - Negative: {neg_count}")

        return True

    except Exception as e:
        print(f"❌ Error organizing {output_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_dataset(name):
    """Verify dataset structure and report statistics."""
    data_dir = Path("D:\\Projects\\EMM_DTI_Replication") / "data" / name

    required_files = ["drugs.csv", "proteins.csv", "interactions.csv"]
    all_exist = all((data_dir / f).exists() for f in required_files)

    if all_exist:
        drugs = pd.read_csv(data_dir / "drugs.csv")
        proteins = pd.read_csv(data_dir / "proteins.csv")
        interactions = pd.read_csv(data_dir / "interactions.csv")

        pos_count = (interactions['interaction'] == 1).sum()
        neg_count = (interactions['interaction'] == 0).sum()

        print(f"\n✅ {name.upper()} Dataset:")
        print(f"   Drugs:        {len(drugs)}")
        print(f"   Proteins:     {len(proteins)}")
        print(f"   Interactions: {len(interactions)}")
        print(f"   Positive:     {pos_count}")
        print(f"   Negative:     {neg_count}")
        print(f"   Balance:      {pos_count/len(interactions)*100:.1f}% positive")

        return True
    else:
        missing = [f for f in required_files if not (data_dir / f).exists()]
        print(f"\n❌ {name.upper()}: Missing {missing}")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("EMM-DTI Multi-Dataset Organization Tool")
    print("="*80)

    results = {}

    # Organize random split datasets
    results['human_random'] = organize_random_split("human_random", "human_random")
    results['celegans_random'] = organize_random_split("celegans_random", "celegans_random")
    results['bindingdb_random'] = organize_random_split("bindingdb_random", "bindingdb_random")

    # Organize single-file datasets
    results['drugbank'] = organize_drugbank_or_enzyme("drugbank", "drugbank")
    results['enzyme'] = organize_drugbank_or_enzyme("enzyme", "enzyme")

    # Verify all datasets
    print("\n" + "="*80)
    print("Dataset Verification Summary")
    print("="*80)

    verify_dataset("human")  # Pre-existing

    if results['human_random']:
        verify_dataset("human_random")
    if results['celegans_random']:
        verify_dataset("celegans_random")
    if results['bindingdb_random']:
        verify_dataset("bindingdb_random")
    if results['drugbank']:
        verify_dataset("drugbank")
    if results['enzyme']:
        verify_dataset("enzyme")

    # Final summary
    print("\n" + "="*80)
    print("Organization Status")
    print("="*80)
    print(f"human:             ✅ Pre-existing")
    print(f"human_random:     {'✅ Done' if results['human_random'] else '❌ Failed'}")
    print(f"celegans_random:   {'✅ Done' if results['celegans_random'] else '❌ Failed'}")
    print(f"bindingdb_random:  {'✅ Done' if results['bindingdb_random'] else '❌ Failed'}")
    print(f"drugbank:          {'✅ Done' if results['drugbank'] else '❌ Failed'}")
    print(f"enzyme:            {'✅ Done' if results['enzyme'] else '❌ Failed'}")
    print("="*80)

    success_count = sum(1 for v in results.values() if v)
    print(f"\nOrganized {success_count}/{len(results)} datasets successfully!")
