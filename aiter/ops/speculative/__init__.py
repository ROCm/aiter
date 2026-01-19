"""
AITER Speculative Decoding Module

Lightweight speculative decoding implementations without framework dependencies.
Provides core algorithms for EAGLE and other speculative methods.

Author: AIter Team
"""

from .eagle_inference import EAGLEInference, EAGLEConfig
from .eagle_utils import (
    organize_draft_results,
    build_tree_structure,
)
from .spec_utils import (
    fast_topk_torch,
    select_top_k_tokens,
    generate_token_bitmask,
)

__all__ = [
    # Main inference class
    'EAGLEInference',
    'EAGLEConfig',
    
    # EAGLE utilities
    'organize_draft_results',
    'build_tree_structure',
    
    # General utilities
    'fast_topk_torch',
    'select_top_k_tokens',
    'generate_token_bitmask',
]

__version__ = '0.1.0'

