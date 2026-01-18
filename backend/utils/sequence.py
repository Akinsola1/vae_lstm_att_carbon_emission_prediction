"""
Sequence creation utilities for time series prediction.

This module creates the sliding window sequences required by LSTM models.
"""

import numpy as np
import pandas as pd


def create_inference_sequence(latent_features: np.ndarray, window_size: int) -> np.ndarray:
    """
    Create a sequence from latent features for LSTM input.
    
    Why sequences are needed:
    - LSTM models process sequences of time steps, not single points
    - window_size defines how many historical time steps to use
    - Shape must be (1, window_size, latent_dim) for single prediction
    
    Training vs Inference:
    - Training: Creates many overlapping sequences from full dataset
    - Inference: Creates ONE sequence from the most recent window_size points
    
    Args:
        latent_features: Array of shape (window_size, latent_dim)
        window_size: Number of time steps in sequence (from config)
    
    Returns:
        Array of shape (1, window_size, latent_dim) ready for model.predict()
    """
    # Verify we have exactly window_size time steps
    # Why: This should already be guaranteed by preprocessing, but verify
    if len(latent_features) != window_size:
        raise ValueError(
            f"Expected {window_size} time steps, got {len(latent_features)}"
        )
    
    # Reshape to (1, window_size, latent_dim)
    # Why the dimensions:
    # - 1: Batch size (predicting one future point)
    # - window_size: Number of historical time steps
    # - latent_dim: Features per time step (3 latent dimensions)
    sequence = latent_features.reshape(1, window_size, latent_features.shape[1])
    
    return sequence


def verify_sequence_shape(sequence: np.ndarray, expected_shape: tuple) -> bool:
    """
    Verify that the sequence shape matches model expectations.
    
    Why this matters:
    - Shape mismatch causes immediate prediction failure
    - Better to catch and report clearly than cryptic TensorFlow error
    
    Args:
        sequence: Created sequence array
        expected_shape: Expected shape (1, window_size, latent_dim)
    
    Returns:
        True if shape matches, raises ValueError otherwise
    """
    if sequence.shape != expected_shape:
        raise ValueError(
            f"Sequence shape mismatch. Expected {expected_shape}, got {sequence.shape}"
        )
    
    return True
