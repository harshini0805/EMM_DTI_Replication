"""
Configuration management for EMM-DTI experiments.

Handles loading, validation, and management of YAML configuration files
with support for command-line overrides.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional, List
from pathlib import Path
import yaml
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatasetConfig:
    """Dataset configuration."""

    name: str = "human"
    data_dir: str = "data/human"
    train_split: float = 0.7
    val_split: float = 0.2
    test_split: float = 0.1
    random_seed: int = 42

    def __post_init__(self) -> None:
        """Validate splits sum to 1.0."""
        total = self.train_split + self.val_split + self.test_split
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Splits must sum to 1.0, got {total}")


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    fcs_embedding_dim: int = 128  # Embedding dimension for FCS fragments
    mamba_hidden_dim: int = 64    # Mamba d_model (standardized to match other 9 architectures)
    mamba_n_layers: int = 2
    mamba_state_size: int = 16
    mamba_expand_factor: int = 2
    cnn_out_channels: int = 3
    cnn_kernel_size: int = 3
    dropout: float = 0.1

    def __post_init__(self) -> None:
        """Validate hyperparameters."""
        if self.fcs_embedding_dim <= 0:
            raise ValueError("fcs_embedding_dim must be positive")
        if self.mamba_hidden_dim <= 0:
            raise ValueError("mamba_hidden_dim must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    device: str = "cuda"
    num_workers: int = 4
    pin_memory: bool = True
    mixed_precision: bool = False

    def __post_init__(self) -> None:
        """Validate training parameters."""
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.device not in ["cuda", "cpu"]:
            raise ValueError("device must be 'cuda' or 'cpu'")


@dataclass
class OptimizationConfig:
    """Optimization parameters."""

    optimizer: str = "adam"
    weight_decay: float = 1e-5
    gradient_clip: float = 1.0
    warmup_steps: int = 0
    lr_scheduler: str = "none"  # "none", "cosine", "linear"

    def __post_init__(self) -> None:
        """Validate optimization parameters."""
        if self.optimizer not in ["adam", "sgd", "adamw"]:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        if self.gradient_clip < 0:
            raise ValueError("gradient_clip must be non-negative")


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_dir: str = "results"
    log_level: str = "INFO"
    tensorboard: bool = True
    wandb: bool = False
    wandb_project: str = "emm-dti"
    save_interval: int = 5

    def __post_init__(self) -> None:
        """Validate logging parameters."""
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log_level: {self.log_level}")


@dataclass
class Config:
    """Main configuration container."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
            ValueError: If configuration validation fails
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        logger.info(f"Loading config from: {path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        Create configuration from dictionary.

        Args:
            data: Configuration dictionary

        Returns:
            Config instance
        """
        config = cls()

        if "dataset" in data:
            config.dataset = DatasetConfig(**data["dataset"])
        if "model" in data:
            config.model = ModelConfig(**data["model"])
        if "training" in data:
            config.training = TrainingConfig(**data["training"])
        if "optimization" in data:
            config.optimization = OptimizationConfig(**data["optimization"])
        if "logging" in data:
            config.logging = LoggingConfig(**data["logging"])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "dataset": asdict(self.dataset),
            "model": asdict(self.model),
            "training": asdict(self.training),
            "optimization": asdict(self.optimization),
            "logging": asdict(self.logging),
        }

    def save(self, path: str | Path) -> None:
        """
        Save configuration to YAML file.

        Args:
            path: Output path for YAML file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving config to: {path}")

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def update_from_args(self, **kwargs) -> None:
        """
        Update configuration from keyword arguments.

        Args:
            **kwargs: Configuration parameters in dotted format
                     e.g., training.batch_size=64, model.dropout=0.2
        """
        for key, value in kwargs.items():
            if value is None:
                continue

            parts = key.split(".")
            if len(parts) == 2:
                section, param = parts
                if hasattr(self, section):
                    section_obj = getattr(self, section)
                    if hasattr(section_obj, param):
                        setattr(section_obj, param, value)
                        logger.info(f"Updated {section}.{param} = {value}")
                    else:
                        logger.warning(f"Unknown parameter: {section}.{param}")
                else:
                    logger.warning(f"Unknown section: {section}")

    def __str__(self) -> str:
        """Pretty string representation."""
        lines = ["=" * 60]
        lines.append("EMM-DTI Configuration")
        lines.append("=" * 60)

        for section_name in ["dataset", "model", "training", "optimization", "logging"]:
            section = getattr(self, section_name)
            lines.append(f"\n[{section_name.upper()}]")
            for key, value in asdict(section).items():
                lines.append(f"  {key}: {value}")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)
