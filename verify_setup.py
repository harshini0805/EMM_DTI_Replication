"""
Comprehensive setup verification script for EMM-DTI.

Checks:
1. Project structure
2. Dependencies
3. Configuration files
4. Data availability & format
5. Model imports & architecture
6. Data pipeline functionality
7. End-to-end sanity check
"""

import sys
from pathlib import Path
from typing import Tuple, List
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s"
)
logger = logging.getLogger(__name__)


class VerificationChecker:
    """Comprehensive setup verification."""

    def __init__(self):
        self.project_root = Path(__file__).parent
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def check(self, condition: bool, message: str, error_msg: str = "") -> bool:
        """Log and track check result."""
        if condition:
            logger.info(f"✓ {message}")
            self.passed += 1
            return True
        else:
            logger.error(f"✗ {message}")
            if error_msg:
                logger.error(f"  → {error_msg}")
            self.failed += 1
            return False

    def warn(self, message: str) -> None:
        """Log warning."""
        logger.warning(f"⚠ {message}")
        self.warnings += 1

    def run_all(self) -> bool:
        """Run all verification checks."""
        logger.info("=" * 80)
        logger.info("EMM-DTI Setup Verification")
        logger.info("=" * 80)

        self.check_project_structure()
        self.check_dependencies()
        self.check_configuration()
        self.check_data()
        self.check_model_imports()
        self.check_data_pipeline()
        self.print_summary()

        return self.failed == 0

    # ===== CHECK 1: Project Structure =====
    def check_project_structure(self) -> None:
        """Verify project directory structure."""
        logger.info("\n[1/6] Checking Project Structure...")

        required_dirs = [
            "emm_dti",
            "emm_dti/models",
            "emm_dti/data",
            "emm_dti/training",
            "emm_dti/utils",
            "configs",
            "data",
        ]

        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            self.check(
                full_path.exists() and full_path.is_dir(),
                f"Directory exists: {dir_path}",
                f"Expected directory not found: {full_path}",
            )

        required_files = [
            "setup.py",
            "pyproject.toml",
            "requirements.txt",
            "README.md",
            "QUICKSTART.md",
            "emm_dti/__init__.py",
            "emm_dti/models/__init__.py",
            "emm_dti/models/mamba_ssm.py",
            "emm_dti/models/fcs.py",
            "emm_dti/models/emm_dti.py",
            "emm_dti/data/__init__.py",
            "emm_dti/data/loaders.py",
            "emm_dti/data/preprocessing.py",
            "emm_dti/training/__init__.py",
            "emm_dti/training/trainer.py",
            "emm_dti/training/metrics.py",
            "emm_dti/utils/__init__.py",
            "emm_dti/utils/config.py",
            "emm_dti/utils/logging_utils.py",
            "emm_dti/utils/device.py",
            "emm_dti/train.py",
            "emm_dti/evaluate.py",
            "configs/default.yaml",
            "configs/train_human.yaml",
        ]

        for file_path in required_files:
            full_path = self.project_root / file_path
            self.check(
                full_path.exists() and full_path.is_file(),
                f"File exists: {file_path}",
                f"Expected file not found: {full_path}",
            )

    # ===== CHECK 2: Dependencies =====
    def check_dependencies(self) -> None:
        """Verify all dependencies are installed."""
        logger.info("\n[2/6] Checking Dependencies...")

        critical_deps = {
            "torch": "PyTorch",
            "numpy": "NumPy",
            "pandas": "Pandas",
            "yaml": "PyYAML",
            "sklearn": "scikit-learn",
            "tqdm": "tqdm",
        }

        for module_name, display_name in critical_deps.items():
            try:
                __import__(module_name)
                self.check(True, f"Dependency installed: {display_name}")
            except ImportError:
                self.check(
                    False,
                    f"Dependency missing: {display_name}",
                    f"Install with: pip install {module_name}",
                )

        # Optional: RDKit
        try:
            import rdkit
            self.check(True, "Optional: RDKit installed (SMILES canonicalization)")
        except ImportError:
            self.warn(
                "Optional: RDKit not installed (SMILES canonicalization disabled). "
                "Install with: pip install rdkit"
            )

    # ===== CHECK 3: Configuration =====
    def check_configuration(self) -> None:
        """Verify configuration files are valid."""
        logger.info("\n[3/6] Checking Configuration Files...")

        config_files = [
            "configs/default.yaml",
            "configs/train_human.yaml",
        ]

        for config_path in config_files:
            full_path = self.project_root / config_path
            if not self.check(full_path.exists(), f"Config file exists: {config_path}"):
                continue

            try:
                import yaml
                with open(full_path) as f:
                    config = yaml.safe_load(f)

                self.check(
                    isinstance(config, dict),
                    f"Config is valid YAML: {config_path}",
                    "Config is not a valid YAML dictionary",
                )

                # Check required sections
                required_sections = ["dataset", "model", "training", "optimization", "logging"]
                for section in required_sections:
                    self.check(
                        section in config,
                        f"  Config section exists: {config_path} → {section}",
                        f"Missing section: {section}",
                    )

            except Exception as e:
                self.check(False, f"Config is parseable: {config_path}", str(e))

    # ===== CHECK 4: Data =====
    def check_data(self) -> None:
        """Verify data files exist and have correct format."""
        logger.info("\n[4/6] Checking Data Files...")

        data_dir = self.project_root / "data" / "human"

        required_data_files = ["drugs.csv", "proteins.csv", "interactions.csv"]

        all_exist = True
        for data_file in required_data_files:
            file_path = data_dir / data_file
            exists = self.check(
                file_path.exists(),
                f"Data file exists: {data_file}",
                f"Expected file at: {file_path}",
            )
            all_exist = all_exist and exists

        if not all_exist:
            self.warn("Data files missing. Please add CSV files to data/human/")
            return

        # Check data format
        try:
            import pandas as pd

            # Check drugs.csv
            drugs_df = pd.read_csv(data_dir / "drugs.csv")
            self.check(
                "drug_id" in drugs_df.columns and "smiles" in drugs_df.columns,
                f"drugs.csv has correct columns (drug_id, smiles)",
                f"Columns found: {list(drugs_df.columns)}",
            )
            self.check(
                len(drugs_df) > 0,
                f"drugs.csv has data ({len(drugs_df)} rows)",
            )

            # Check proteins.csv
            proteins_df = pd.read_csv(data_dir / "proteins.csv")
            self.check(
                "protein_id" in proteins_df.columns and "sequence" in proteins_df.columns,
                f"proteins.csv has correct columns (protein_id, sequence)",
                f"Columns found: {list(proteins_df.columns)}",
            )
            self.check(
                len(proteins_df) > 0,
                f"proteins.csv has data ({len(proteins_df)} rows)",
            )

            # Check interactions.csv
            interactions_df = pd.read_csv(data_dir / "interactions.csv")
            required_cols = ["drug_id", "protein_id", "interaction"]
            self.check(
                all(col in interactions_df.columns for col in required_cols),
                f"interactions.csv has correct columns",
                f"Expected {required_cols}, got {list(interactions_df.columns)}",
            )
            self.check(
                len(interactions_df) > 0,
                f"interactions.csv has data ({len(interactions_df)} rows)",
            )

            # Summary
            logger.info(
                f"  Summary: {len(drugs_df)} drugs × {len(proteins_df)} proteins "
                f"= {len(interactions_df)} interactions"
            )

        except Exception as e:
            self.check(False, "Data files are valid CSV", str(e))

    # ===== CHECK 5: Model Imports =====
    def check_model_imports(self) -> None:
        """Verify all model components can be imported."""
        logger.info("\n[5/6] Checking Model Imports...")

        try:
            from emm_dti.models.mamba_ssm import BidirectionalMambaSSM
            self.check(True, "Import: BidirectionalMambaSSM")
        except Exception as e:
            self.check(False, "Import: BidirectionalMambaSSM", str(e))

        try:
            from emm_dti.models.fcs import FCSModule, FragmentVocabulary
            self.check(True, "Import: FCSModule, FragmentVocabulary")
        except Exception as e:
            self.check(False, "Import: FCSModule", str(e))

        try:
            from emm_dti.models.emm_dti import EMMDTI
            self.check(True, "Import: EMMDTI model")
        except Exception as e:
            self.check(False, "Import: EMMDTI", str(e))

        try:
            from emm_dti.data.loaders import DTIDataset, DTIDataModule
            self.check(True, "Import: DTIDataset, DTIDataModule")
        except Exception as e:
            self.check(False, "Import: Data loaders", str(e))

        try:
            from emm_dti.training.trainer import Trainer
            from emm_dti.training.metrics import Metrics
            self.check(True, "Import: Trainer, Metrics")
        except Exception as e:
            self.check(False, "Import: Training modules", str(e))

        try:
            from emm_dti.utils.config import Config
            from emm_dti.utils.device import get_device
            self.check(True, "Import: Config, device utilities")
        except Exception as e:
            self.check(False, "Import: Utilities", str(e))

    # ===== CHECK 6: Data Pipeline =====
    def check_data_pipeline(self) -> None:
        """End-to-end sanity check of data pipeline."""
        logger.info("\n[6/6] Checking Data Pipeline...")

        data_dir = self.project_root / "data" / "human"

        if not (data_dir / "drugs.csv").exists():
            self.warn("Skipping data pipeline check (data files missing)")
            return

        try:
            from emm_dti.data.loaders import DTIDataModule

            # Initialize data module
            dm = DTIDataModule(data_dir=str(data_dir))
            self.check(True, "DTIDataModule initialized successfully")

            # Check vocabulary
            self.check(
                len(dm.fcs_vocab) > 0,
                f"FCS vocabulary created ({len(dm.fcs_vocab)} tokens)",
            )

            # Create loaders
            train_loader, val_loader, test_loader = dm.create_loaders(batch_size=8)
            self.check(True, "DataLoaders created successfully")

            # Test single batch
            drug_idx, protein_idx, labels = next(iter(train_loader))
            self.check(
                drug_idx.shape[0] == 8,
                f"Batch processing works (batch_size=8)",
                f"Got batch_size={drug_idx.shape[0]}",
            )

            # Test model forward pass
            import torch
            from emm_dti.models.emm_dti import EMMDTI

            device = torch.device("cpu")  # Use CPU for verification
            model = EMMDTI(vocab_size=len(dm.fcs_vocab))
            model = model.to(device)

            drug_idx = drug_idx.to(device)
            protein_idx = protein_idx.to(device)

            with torch.no_grad():
                predictions = model(drug_idx, protein_idx)

            self.check(
                predictions.shape == (8, 1),
                f"Model forward pass works (output shape: {predictions.shape})",
                f"Expected (8, 1), got {predictions.shape}",
            )

            probs = torch.sigmoid(predictions)
            self.check(
                (probs >= 0).all() and (probs <= 1).all(),
                "Model outputs valid logit predictions (convertible to 0-1 probabilities)",
            )

        except Exception as e:
            self.check(False, "Data pipeline sanity check", f"{type(e).__name__}: {e}")

    # ===== Summary =====
    def print_summary(self) -> None:
        """Print verification summary."""
        total = self.passed + self.failed
        percentage = (self.passed / total * 100) if total > 0 else 0

        logger.info("\n" + "=" * 80)
        logger.info("Verification Summary")
        logger.info("=" * 80)
        logger.info(f"Passed:  {self.passed}")
        logger.info(f"Failed:  {self.failed}")
        if self.warnings > 0:
            logger.info(f"Warnings: {self.warnings}")
        logger.info(f"Success Rate: {percentage:.1f}%")
        logger.info("=" * 80)

        if self.failed == 0:
            logger.info("\n✓ All checks passed! You're ready to train.")
            logger.info("\nTo start training:")
            logger.info("  python -m emm_dti.train --config configs/train_human.yaml")
        else:
            logger.error("\n✗ Some checks failed. Please fix the issues above.")
            logger.error(
                "\nCommon fixes:"
            )
            logger.error("  1. Install missing packages: pip install -e .")
            logger.error("  2. Add data files to data/human/")
            logger.error("  3. Verify data format matches documentation")

        logger.info("=" * 80 + "\n")


def main():
    """Run verification."""
    checker = VerificationChecker()
    success = checker.run_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
