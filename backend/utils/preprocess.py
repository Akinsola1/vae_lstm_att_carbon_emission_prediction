"""
Preprocessing utilities for backend inference.

This module handles data transformation to match the exact preprocessing
pipeline used during training. Any deviation will cause incorrect predictions.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def validate_csv_data(df: pd.DataFrame, required_columns: list) -> Tuple[bool, str]:
    """
    Validate that CSV contains required columns and sufficient data.
    
    Why: Ensures input data matches training data schema.
    Missing columns or wrong names will cause prediction failures.
    
    Args:
        df: Input DataFrame from CSV
        required_columns: List of column names expected (from config)
    
    Returns:
        (is_valid, error_message) tuple
    """
    missing_cols = set(required_columns) - set(df.columns)
    
    if missing_cols:
        return False, f"Missing required columns: {missing_cols}"
    
    if df.empty:
        return False, "CSV file is empty"
    
    return True, ""


def preprocess_input_data(
    df: pd.DataFrame,
    required_columns: list,
    window_size: int
) -> Tuple[pd.DataFrame, str]:
    """
    Preprocess uploaded CSV to match training data format.
    
    Why each step:
    - Sort by year: Time series data must be chronologically ordered
    - Drop nulls: Model was trained on complete records only
    - Keep required columns: Remove irrelevant features
    - Extract last N years: We need exactly window_size historical points
    
    Args:
        df: Raw DataFrame from CSV upload
        required_columns: Features used during training
        window_size: Number of historical years needed (from config)
    
    Returns:
        (processed_df, error_message) tuple
    """
    # Sort by year to ensure chronological order
    # Why: LSTM models are sensitive to time ordering
    df = df.sort_values(by="year").reset_index(drop=True)
    
    # Keep only the columns used during training
    # Why: Extra columns will break the model input shape
    df = df[required_columns]
    
    # Remove rows with missing values
    # Why: Model was trained on complete data; NaNs will cause errors
    df = df.dropna().reset_index(drop=True)
    
    # Check if we have enough historical data
    # Why: We need at least window_size years to create a sequence
    if len(df) < window_size:
        return None, f"Need at least {window_size} years of data, got {len(df)}"
    
    # Extract the most recent window_size years
    # Why: We predict the NEXT year after the most recent window
    df = df.tail(window_size).reset_index(drop=True)
    
    return df, ""


def scale_features(df: pd.DataFrame, scaler) -> np.ndarray:
    """
    Apply the same MinMaxScaler used during training.
    
    Why critical:
    - Model was trained on 0-1 scaled data
    - Using a different scale would make predictions meaningless
    - Must use the EXACT scaler from training (loaded from scaler.pkl)
    
    Args:
        df: DataFrame with features (excluding year)
        scaler: Loaded MinMaxScaler object from training
    
    Returns:
        Scaled numpy array
    """
    # Remove 'year' column before scaling
    # Why: 'year' is not a feature for prediction, just an index
    features = df.drop(columns=["year"])
    
    # Apply the saved scaler
    # Why: transform() applies the min/max learned during training
    # NEVER use fit_transform() in inference - that would learn new min/max
    scaled_data = scaler.transform(features.values)
    
    return scaled_data


def encode_to_latent_space(scaled_data: np.ndarray, encoder_model) -> np.ndarray:
    """
    Transform features to VAE latent space representation.
    
    Why use VAE encoding:
    - Reduces dimensionality (7 features -> 3 latent dimensions)
    - Captures non-linear relationships between features
    - The LSTM-Attention model was trained on latent features, not raw features
    
    Args:
        scaled_data: Scaled feature array
        encoder_model: Trained VAE encoder (loaded from vae_encoder.h5)
    
    Returns:
        Latent space representation
    """
    # Encode each time step through the VAE encoder
    # Why: Model expects latent representations, not raw scaled features
    latent_features = encoder_model.predict(scaled_data, verbose=0)
    
    return latent_features
