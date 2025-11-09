"""
TinyStories Model Package

This package contains the model architecture implementation following the
TinyStories paper specifications.
"""

from model.model import Config, Model, MultiHeadAttention, FFN, TransformerBlock

__all__ = ['Config', 'Model', 'MultiHeadAttention', 'FFN', 'TransformerBlock']

