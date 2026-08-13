"""
Bidirectional Mamba using official mamba-ssm library.

This is the "proper" Mamba implementation using the official Selective SSM,
as described in the paper: "based on the Selective StateSpaceModel (SSM)"
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)

try:
    from mamba_ssm import Mamba
    HAS_MAMBA_SSM = True
except ImportError:
    HAS_MAMBA_SSM = False
    logger.warning("mamba-ssm not installed. Install with: pip install mamba-ssm")


class BidirectionalMambaSSM(nn.Module):
    """
    Bidirectional Mamba using official mamba-ssm library.

    Uses the Selective StateSpaceModel (SSM) as described in the paper.
    Processes sequences in both directions for bidirectional information.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        state_size: int = 16,
        dropout: float = 0.0,
        n_layers: int = 1,
    ):
        """
        Initialize bidirectional Mamba with official mamba-ssm.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Output hidden dimension per direction
            state_size: State space dimension (d_state in Mamba)
            dropout: Dropout probability
            n_layers: Number of stacked Mamba layers
        """
        super().__init__()

        if not HAS_MAMBA_SSM:
            raise ImportError(
                "mamba-ssm is required. Install with: pip install mamba-ssm"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.state_size = state_size

        # Project input to hidden_dim if needed
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else None

        # Forward Mamba layers
        self.forward_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=hidden_dim,
                    d_state=state_size,
                    d_conv=4,  # Conv1D kernel size
                    expand=2,  # Expansion factor
                    dt_rank="auto",
                    dt_min=0.001,
                    dt_max=0.1,
                    dt_init="random",
                    dt_scale=1.0,
                    bias=True,
                    conv_bias=True,
                    use_fast_path=True,
                )
                for i in range(n_layers)
            ]
        )

        # Backward Mamba layers (separate parameters)
        self.backward_layers = nn.ModuleList(
            [
                Mamba(
                    d_model=hidden_dim,
                    d_state=state_size,
                    d_conv=4,
                    expand=2,
                    dt_rank="auto",
                    dt_min=0.001,
                    dt_max=0.1,
                    dt_init="random",
                    dt_scale=1.0,
                    bias=True,
                    conv_bias=True,
                    use_fast_path=True,
                )
                for i in range(n_layers)
            ]
        )

        # Output projection: concatenates forward + backward → hidden_dim
        self.output_proj = nn.Linear(2 * hidden_dim, hidden_dim)

        logger.info(
            f"BidirectionalMambaSSM initialized: "
            f"input_dim={input_dim}, hidden_dim={hidden_dim}, "
            f"d_state={state_size}, n_layers={n_layers}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Bidirectional forward pass using official Mamba-SSM.

        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)

        Returns:
            Output tensor of shape (batch, seq_len, hidden_dim)
        """
        batch_size, seq_len, _ = x.shape

        # Project input if needed
        if self.input_proj is not None:
            x = self.input_proj(x)  # (batch, seq_len, hidden_dim)

        # Forward direction
        fwd = x
        for layer in self.forward_layers:
            fwd = layer(fwd)  # (batch, seq_len, hidden_dim)

        # Backward direction
        # Flip sequence, process, flip back
        bwd = torch.flip(x, [1])  # Reverse sequence
        for layer in self.backward_layers:
            bwd = layer(bwd)
        bwd = torch.flip(bwd, [1])  # Reverse back to original order

        # Concatenate forward + backward representations
        combined = torch.cat([fwd, bwd], dim=-1)  # (batch, seq_len, 2*hidden_dim)

        # Project to final dimension
        output = self.output_proj(combined)  # (batch, seq_len, hidden_dim)

        return output

    def extra_repr(self) -> str:
        """String representation of layer configuration."""
        return (
            f"input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"state_size={self.state_size}, n_layers={self.n_layers}"
        )
