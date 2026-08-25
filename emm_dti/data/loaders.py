import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Set, Generator
import pandas as pd
import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold

from emm_dti.data.preprocessing import DataPreprocessor
from emm_dti.models.fcs import FCSModule, FragmentVocabulary

logger = logging.getLogger(__name__)


class DTIDataset(Dataset):
    """
    PyTorch Dataset for DTI prediction.

    Handles loading and indexing of drug-protein interaction pairs.
    Uses FCS-mined patterns for sequence representation.
    """

    def __init__(
        self,
        drug_sequences: List[str],
        protein_sequences: List[str],
        interactions: List[int],
        fcs_vocab: FragmentVocabulary,
        fcs_patterns: Dict[int, set] | None = None,
        max_drug_len: int = 100,
        max_protein_len: int = 200,
    ):
        """
        Initialize DTI dataset.

        Args:
            drug_sequences: List of SMILES strings
            protein_sequences: List of protein sequences
            interactions: List of binary interaction labels
            fcs_vocab: Fragment vocabulary for encoding
            fcs_patterns: FCS mined patterns by length (from FCSModule)
            max_drug_len: Maximum drug sequence length
            max_protein_len: Maximum protein sequence length

        Raises:
            ValueError: If sequence lengths don't match
        """
        if not (len(drug_sequences) == len(protein_sequences) == len(interactions)):
            raise ValueError("Sequence and label counts must match")

        self.drug_sequences = drug_sequences
        self.protein_sequences = protein_sequences
        self.interactions = torch.tensor(interactions, dtype=torch.float32)
        self.fcs_vocab = fcs_vocab
        self.fcs_patterns = fcs_patterns or {}
        self.max_drug_len = max_drug_len
        self.max_protein_len = max_protein_len

        # Build pattern list ordered by length (longest first for greedy matching)
        self.patterns_by_len = []
        for length in sorted(self.fcs_patterns.keys(), reverse=True):
            self.patterns_by_len.extend(self.fcs_patterns[length])

        logger.info(
            f"Initialized DTIDataset with {len(self)} samples "
            f"(vocab_size={len(fcs_vocab)}, fcs_patterns={len(self.patterns_by_len)})"
        )

    def __len__(self) -> int:
        """Dataset size."""
        return len(self.interactions)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (drug_indices, protein_indices, interaction_label)
        """
        # Get sequences
        drug_seq = self.drug_sequences[idx]
        protein_seq = self.protein_sequences[idx]
        label = self.interactions[idx]

        # Convert to fragment indices
        drug_indices = self._sequence_to_indices(drug_seq, self.max_drug_len)
        protein_indices = self._sequence_to_indices(protein_seq, self.max_protein_len)

        return drug_indices, protein_indices, label

    def _sequence_to_indices(self, sequence: str, max_len: int) -> torch.Tensor:
        """
        Convert sequence to FCS fragment indices using greedy matching.

        Uses FCS-mined frequent patterns for sequence tokenization.
        Falls back to character-level for unmatched regions.

        Args:
            sequence: Input sequence
            max_len: Maximum length (pads or truncates)

        Returns:
            Tensor of fragment indices
        """
        indices = []
        i = 0
        sequence = sequence[:max_len]

        while i < len(sequence) and len(indices) < max_len:
            matched = False

            # Greedy matching: try longest patterns first
            for pattern in self.patterns_by_len:
                if sequence[i:].startswith(pattern):
                    indices.append(self.fcs_vocab.get_index(pattern))
                    i += len(pattern)
                    matched = True
                    break

            # If no pattern matched, use character-level (1-mer)
            if not matched:
                indices.append(self.fcs_vocab.get_index(sequence[i]))
                i += 1

        # Pad to max_len
        if len(indices) < max_len:
            indices += [0] * (max_len - len(indices))
        else:
            indices = indices[:max_len]

        return torch.tensor(indices, dtype=torch.long)


class DTIDataModule:
    """
    Data module for managing DTI dataset splits and loading.

    Handles train/val/test splitting and DataLoader creation.
    """

    def __init__(
        self,
        data_dir: str | Path,
        train_split: float = 0.7,
        val_split: float = 0.2,
        test_split: float = 0.1,
        random_seed: int = 42,
    ):
        """
        Initialize data module.

        Args:
            data_dir: Directory containing data files
            train_split: Training set fraction
            val_split: Validation set fraction
            test_split: Test set fraction
            random_seed: Random seed for reproducibility

        Raises:
            FileNotFoundError: If data files not found
            ValueError: If splits don't sum to 1.0
        """
        self.data_dir = Path(data_dir)
        self.random_seed = random_seed

        # Validate splits
        total = train_split + val_split + test_split
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Splits must sum to 1.0, got {total}")

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split

        # Check files exist
        required_files = ["drugs.csv", "proteins.csv", "interactions.csv"]
        for filename in required_files:
            filepath = self.data_dir / filename
            if not filepath.exists():
                raise FileNotFoundError(f"Required file not found: {filepath}")

        # Load data
        self._load_data()

        # Check if pre-split benchmark column exists
        if "split" in self.interactions_df.columns:
            logger.info("Found pre-defined 'split' column in interactions.csv - preserving benchmark splits.")
            train_interactions_df = self.interactions_df[self.interactions_df["split"] == "train"]
        else:
            # Split interactions BEFORE mining FCS (prevent data leakage)
            total_samples = len(self.interactions_df)
            train_size = int(total_samples * self.train_split)
            train_interactions_df = self.interactions_df.iloc[:train_size]

        # Create FCS vocabulary from TRAINING data only
        self._create_fcs_vocabulary(train_interactions_df=train_interactions_df)

        logger.info(f"Data module initialized with {len(self)} total samples")

    def _load_data(self) -> None:
        """Load data from CSV files."""
        logger.info(f"Loading data from {self.data_dir}")

        # Load drugs
        drugs_df = pd.read_csv(self.data_dir / "drugs.csv")
        if "smiles" not in drugs_df.columns:
            raise ValueError("drugs.csv must contain 'smiles' column")
        self.drugs = drugs_df[["drug_id", "smiles"]].set_index("drug_id").to_dict()["smiles"]

        # Load proteins
        proteins_df = pd.read_csv(self.data_dir / "proteins.csv")
        if "sequence" not in proteins_df.columns:
            raise ValueError("proteins.csv must contain 'sequence' column")
        self.proteins = (
            proteins_df[["protein_id", "sequence"]].set_index("protein_id").to_dict()["sequence"]
        )

        # Load interactions
        interactions_df = pd.read_csv(self.data_dir / "interactions.csv")
        required_cols = ["drug_id", "protein_id", "interaction"]
        if not all(col in interactions_df.columns for col in required_cols):
            raise ValueError(f"interactions.csv must contain columns: {required_cols}")

        self.interactions_df = interactions_df

        logger.info(
            f"Loaded {len(self.drugs)} drugs, {len(self.proteins)} proteins, "
            f"{len(self.interactions_df)} interactions"
        )

    def _create_fcs_vocabulary(self, train_interactions_df: Optional['pd.DataFrame'] = None) -> None:
        """
        Create FCS vocabulary from training drug and protein sequences ONLY.

        Args:
            train_interactions_df: Training interactions to mine patterns from.
                                   If None, uses all interactions (legacy behavior).
        """
        # Optional: try to load cached patterns for reproducibility
        cache_path = self.data_dir / ".fcs_cache.pkl"

        if cache_path.exists():
            logger.info(f"Loading cached FCS patterns from {cache_path}")
            try:
                import pickle
                with open(cache_path, "rb") as f:
                    self.fcs = pickle.load(f)

                self.fcs_vocab = FragmentVocabulary()
                self.fcs_vocab.build_from_fcs(self.fcs)
                logger.info(f"Loaded {len(self.fcs_vocab)} patterns from cache")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}. Re-mining patterns...")

        logger.info("Mining FCS patterns from TRAINING sequences only...")

        # Mine from training data only to prevent data leakage
        if train_interactions_df is not None:
            train_drugs = train_interactions_df["drug_id"].unique()
            train_proteins = train_interactions_df["protein_id"].unique()

            drug_sequences = [self.drugs[drug_id] for drug_id in train_drugs]
            protein_sequences = [self.proteins[protein_id] for protein_id in train_proteins]
            logger.info(f"Mining from {len(train_drugs)} training drugs + {len(train_proteins)} training proteins (Separately)")
        else:
            # Fallback to all sequences (legacy)
            logger.warning("No training split provided - using all sequences for FCS mining (data leakage!)")
            drug_sequences = list(self.drugs.values())
            protein_sequences = list(self.proteins.values())

        # Mine frequent patterns separately to prevent modality suppression
        fcs_drugs = FCSModule(min_support=0.3)
        fcs_proteins = FCSModule(min_support=0.3)
        
        drug_patterns = fcs_drugs.mine(drug_sequences, max_k=3)
        protein_patterns = fcs_proteins.mine(protein_sequences, max_k=3)
        
        self.fcs = FCSModule(min_support=0.3)
        from collections import defaultdict
        combined_patterns = defaultdict(set)
        for k, p in drug_patterns.items():
            combined_patterns[k].update(p)
        for k, p in protein_patterns.items():
            combined_patterns[k].update(p)
            
        self.fcs.frequent_patterns = combined_patterns
        self.fcs.support_counts = {**fcs_drugs.support_counts, **fcs_proteins.support_counts}
        patterns = combined_patterns

        logger.info(f"FCS mining complete:")
        for k, p in patterns.items():
            logger.info(f"  {k}-mers: {len(p)} patterns")

        # Build vocabulary from mined patterns
        self.fcs_vocab = FragmentVocabulary()
        self.fcs_vocab.build_from_fcs(self.fcs)

        logger.info(f"Built vocabulary with {len(self.fcs_vocab)} tokens from FCS patterns")

        # Optional: cache patterns for reproducibility
        try:
            import pickle
            with open(cache_path, "wb") as f:
                pickle.dump(self.fcs, f)
            logger.info(f"Cached FCS patterns to {cache_path}")
        except Exception as e:
            logger.warning(f"Failed to cache patterns: {e}")

    def __len__(self) -> int:
        """Total number of samples."""
        return len(self.interactions_df)

    def _create_dataset_from_df(self, df: pd.DataFrame) -> DTIDataset:
        """Helper to create a DTIDataset from a specific DataFrame slice."""
        drug_seqs = [self.drugs[drug_id] for drug_id in df["drug_id"]]
        protein_seqs = [self.proteins[protein_id] for protein_id in df["protein_id"]]
        interactions = df["interaction"].tolist()

        return DTIDataset(
            drug_sequences=drug_seqs,
            protein_sequences=protein_seqs,
            interactions=interactions,
            fcs_vocab=self.fcs_vocab,
            fcs_patterns=self.fcs.get_patterns(),
        )

    def create_loaders(
        self,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test DataLoaders.

        Args:
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for faster GPU transfer

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if "split" in self.interactions_df.columns:
            logger.info("Creating DataLoaders from pre-defined benchmark splits...")
            train_df = self.interactions_df[self.interactions_df["split"] == "train"]
            val_df = self.interactions_df[self.interactions_df["split"].isin(["valid", "val"])]
            test_df = self.interactions_df[self.interactions_df["split"] == "test"]

            train_dataset = self._create_dataset_from_df(train_df)
            val_dataset = self._create_dataset_from_df(val_df)
            test_dataset = self._create_dataset_from_df(test_df)
        else:
            logger.info("Creating DataLoaders via random split...")
            full_dataset = self._create_dataset_from_df(self.interactions_df)
            total_size = len(full_dataset)
            train_size = int(total_size * self.train_split)
            val_size = int(total_size * self.val_split)
            test_size = total_size - train_size - val_size

            train_dataset, val_dataset, test_dataset = random_split(
                full_dataset,
                [train_size, val_size, test_size],
                generator=torch.Generator().manual_seed(self.random_seed),
            )

        logger.info(
            f"Loaded splits: train={len(train_dataset):,}, "
            f"val={len(val_dataset):,}, test={len(test_dataset):,}"
        )

        # Create loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return train_loader, val_loader, test_loader

    def get_stratified_folds(
        self,
        num_folds: int = 5,
        random_state: int = 42,
    ) -> Generator[Tuple[List[int], List[int], List[int]], None, None]:
        """
        Generate stratified k-fold indices for cross-validation.

        Yields:
            Tuples of (train_indices, val_indices, test_indices) for each fold
        """
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)

        # Get labels for stratification
        interactions = self.interactions_df["interaction"].values

        for fold_idx, (temp_idx, test_idx) in enumerate(skf.split(
            self.interactions_df, interactions
        )):
            # Split remaining data into train/val (80/20 of temp)
            temp_interactions = interactions[temp_idx]

            skf_inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
            for train_idx_rel, val_idx_rel in skf_inner.split(
                pd.DataFrame(temp_interactions), temp_interactions
            ):
                # Map back to original indices
                train_idx = temp_idx[train_idx_rel].tolist()
                val_idx = temp_idx[val_idx_rel].tolist()
                test_idx_list = test_idx.tolist()

                yield train_idx, val_idx, test_idx_list
                break  # Only use first split for CV (80/20)

    def create_fold_loader(
        self,
        fold_indices: List[int],
        batch_size: int = 16,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> DataLoader:
        """
        Create a DataLoader for a specific fold.

        Args:
            fold_indices: Indices of samples in this fold
            batch_size: Batch size
            num_workers: Number of data loading workers
            pin_memory: Whether to pin memory for GPU transfer

        Returns:
            DataLoader for the fold
        """
        fold_df = self.interactions_df.iloc[fold_indices]
        fold_dataset = self._create_dataset_from_df(fold_df)

        loader = DataLoader(
            fold_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        return loader
