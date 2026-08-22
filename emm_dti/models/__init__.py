"""Model implementations for EMM-DTI.

Paper Faithful Reproduction Components:
✅ FCS: Frequent Continuous Subsequence mining
✅ EMMDTI: Main model (FCS → Embedding → Mamba-SSM → CNN → MLP)
✅ BidirectionalMambaSSM: Official Selective StateSpaceModel
"""

from emm_dti.models.fcs import FCSModule, FragmentVocabulary
from emm_dti.models.mamba_ssm import BidirectionalMambaSSM
from emm_dti.models.emm_dti import EMMDTI

__all__ = [
    "FCSModule",
    "FragmentVocabulary",
    "BidirectionalMambaSSM",
    "EMMDTI",
]
