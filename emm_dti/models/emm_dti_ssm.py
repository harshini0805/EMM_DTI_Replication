"""
EMM-DTI with Official Mamba-SSM (Selective StateSpaceModel).

This version uses the official mamba-ssm library for the Selective SSM,
as explicitly mentioned in the paper.

Separate from emm_dti.py to allow side-by-side comparison.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional
import logging

from emm_dti.models.fcs import FCSModule, FragmentVocabulary
from emm_dti.models.mamba_ssm import BidirectionalMambaSSM

logger = logging.getLogger(__name__)


class EMMDTI_SSM(nn.Module):
    """
    Enhanced Mamba-based Model for Drug-Target Interaction Prediction.

    Uses OFFICIAL Mamba-SSM (Selective StateSpaceModel) library.

    Architecture:
        1. FCS Module: Decomposes SMILES/proteins into substructures
        2. Embedding: Fragment embeddings + layer normalization
        3. Mamba-SSM: Bidirectional selective state-space sequence processing
        4. Interaction: 2D matrix via dot product
        5. CNN: Convolutional feature extraction
        6. Predictor: MLP binary classifier
    """

    def __init__(
        self,
        vocab_size: int,
        fcs_embedding_dim: int = 128,
        mamba_hidden_dim: int = 64,  # Standardized to match other 9 architectures
        mamba_n_layers: int = 2,
        mamba_state_size: int = 16,
        mamba_expand_factor: int = 2,
        cnn_out_channels: int = 3,
        cnn_kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        """
        Initialize EMM-DTI with official Mamba-SSM.

        Args:
            vocab_size: Size of fragment vocabulary
            fcs_embedding_dim: Dimension of fragment embeddings
            mamba_hidden_dim: Hidden dimension of Mamba layers
            mamba_n_layers: Number of Mamba layers
            mamba_state_size: State space dimension (d_state)
            mamba_expand_factor: Expansion factor (ignored, Mamba uses d_conv/expand)
            cnn_out_channels: Number of output channels for Conv2D
            cnn_kernel_size: Kernel size for Conv2D
            dropout: Dropout probability
        """
        super().__init__()

        self.vocab_size = vocab_size
        self.fcs_embedding_dim = fcs_embedding_dim
        self.mamba_hidden_dim = mamba_hidden_dim

        # ===== Embedding Layer =====
        self.embedding = nn.Embedding(vocab_size, fcs_embedding_dim, padding_idx=0)
        self.embedding_norm = nn.LayerNorm(fcs_embedding_dim)

        # ===== Drug Processing with Mamba-SSM =====
        self.drug_mamba = BidirectionalMambaSSM(
            input_dim=fcs_embedding_dim,
            hidden_dim=mamba_hidden_dim,
            state_size=mamba_state_size,
            dropout=dropout,
            n_layers=mamba_n_layers,
        )

        # ===== Protein Processing with Mamba-SSM =====
        self.protein_mamba = BidirectionalMambaSSM(
            input_dim=fcs_embedding_dim,
            hidden_dim=mamba_hidden_dim,
            state_size=mamba_state_size,
            dropout=dropout,
            n_layers=mamba_n_layers,
        )

        # ===== Interaction Matrix CNN =====
        # Input: (batch, 1, drug_len, protein_len)
        # After conv: (batch, cnn_out_channels, drug_len, protein_len)
        self.interaction_cnn = nn.Sequential(
            nn.Conv2d(1, cnn_out_channels, kernel_size=cnn_kernel_size, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d((1, 1)),
        )

        # ===== Predictor MLP =====
        mlp_input_dim = cnn_out_channels + 2 * mamba_hidden_dim
        self.predictor = nn.Sequential(
            nn.Linear(mlp_input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            # Note: No Sigmoid here - BCEWithLogitsLoss handles it
        )

        # Initialize weights
        self._init_weights()

        logger.info(
            f"EMM-DTI-SSM model initialized with {self._count_parameters()} parameters "
            f"(using official Mamba-SSM library)"
        )

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        drug_indices: torch.Tensor,
        protein_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through EMM-DTI-SSM model.

        Args:
            drug_indices: Drug fragment indices, shape (batch, drug_seq_len)
            protein_indices: Protein fragment indices, shape (batch, protein_seq_len)

        Returns:
            DTI prediction logits, shape (batch, 1)
        """
        batch_size = drug_indices.size(0)

        # ===== Embedding =====
        drug_emb = self.embedding(drug_indices)  # (batch, drug_len, emb_dim)
        drug_emb = self.embedding_norm(drug_emb)

        protein_emb = self.embedding(protein_indices)  # (batch, protein_len, emb_dim)
        protein_emb = self.embedding_norm(protein_emb)

        # ===== Mamba-SSM Processing =====
        # Uses official Selective StateSpaceModel
        drug_repr = self.drug_mamba(drug_emb)  # (batch, drug_len, hidden_dim)
        protein_repr = self.protein_mamba(protein_emb)  # (batch, protein_len, hidden_dim)

        # ===== Pooling for global representation =====
        drug_pool = drug_repr.mean(dim=1)  # (batch, hidden_dim)
        protein_pool = protein_repr.mean(dim=1)  # (batch, hidden_dim)

        # ===== Interaction Matrix =====
        # Compute outer product: (batch, drug_len, protein_len)
        interaction_matrix = torch.bmm(drug_repr, protein_repr.transpose(1, 2))
        interaction_matrix = interaction_matrix.unsqueeze(1)  # (batch, 1, drug_len, protein_len)

        # ===== CNN Feature Extraction =====
        cnn_features = self.interaction_cnn(interaction_matrix)  # (batch, cnn_out, 1, 1)
        cnn_features = cnn_features.view(batch_size, -1)  # (batch, cnn_out)

        # ===== Concatenate and Predict =====
        final_features = torch.cat([cnn_features, drug_pool, protein_pool], dim=1)
        prediction = self.predictor(final_features)  # (batch, 1)

        return prediction

    def load_checkpoint(self, path: str) -> None:
        """
        Load model from checkpoint.

        Args:
            path: Path to checkpoint file
        """
        device = next(self.parameters()).device
        checkpoint = torch.load(path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded checkpoint from {path}")

    def save_checkpoint(self, path: str, optimizer_state: Optional[Dict] = None) -> None:
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            optimizer_state: Optional optimizer state dict
        """
        checkpoint = {"model_state_dict": self.state_dict()}
        if optimizer_state:
            checkpoint["optimizer_state_dict"] = optimizer_state

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def get_model_info(self) -> Dict[str, any]:
        """
        Get model architecture information.

        Returns:
            Dictionary with model configuration
        """
        return {
            "vocab_size": self.vocab_size,
            "fcs_embedding_dim": self.fcs_embedding_dim,
            "mamba_hidden_dim": self.mamba_hidden_dim,
            "total_parameters": self._count_parameters(),
            "device": next(self.parameters()).device.type,
            "mamba_type": "Official Mamba-SSM (Selective StateSpaceModel)",
        }

    def __str__(self) -> str:
        """String representation of model."""
        total_params = self._count_parameters()
        return (
            f"EMM-DTI-SSM(\n"
            f"  vocab_size={self.vocab_size},\n"
            f"  fcs_embedding_dim={self.fcs_embedding_dim},\n"
            f"  mamba_hidden_dim={self.mamba_hidden_dim},\n"
            f"  total_parameters={total_params},\n"
            f"  mamba_type='Official Mamba-SSM (Selective)'\n"
            f")"
        )
