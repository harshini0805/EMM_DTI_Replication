"""
EMM-DTI: Enhanced Mamba-Based Model for Drug-Target Interaction Prediction.

A PyTorch implementation for predicting drug-target interactions using
Mamba-based sequence models with Frequent Continuous Subsequence (FCS) mining.
"""

__version__ = "0.1.0"
__author__ = "EMM-DTI Development Team"
__all__ = [
    "EMMDTI",
    "FCSModule",
    "MambaLayer",
    "Config",
    "setup_logging",
]

from emm_dti.models.emm_dti import EMMDTI
from emm_dti.models.fcs import FCSModule
from emm_dti.models.mamba import MambaLayer
from emm_dti.utils.config import Config
from emm_dti.utils.logging_utils import setup_logging
