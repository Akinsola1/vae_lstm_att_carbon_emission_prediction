"""
FastAPI Backend for VAE-LSTM-Attention GHG Emission Prediction.

This is an INFERENCE-ONLY service. No training occurs here.
All models are pre-trained and loaded at startup.

Architecture:
- Models loaded once at startup (not per request)
- CSV upload endpoint processes and returns predictions
- All preprocessing matches training pipeline exactly
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import joblib
# TensorFlow import
# In TensorFlow 2.16+, Keras v3 is the default and should be imported directly
import tensorflow as tf
import keras

# Enable unsafe deserialization to allow loading Lambda layers
# We trust the local models we just generated
keras.config.enable_unsafe_deserialization()

from keras import layers

@keras.saving.register_keras_serializable()
class SumLayer(layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)
        
    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

# import keras
import io
from typing import Dict

# Import our utility modules
from utils.preprocess import (
    validate_csv_data,
    preprocess_input_data,
    scale_features,
    encode_to_latent_space
)
from utils.sequence import create_inference_sequence, verify_sequence_shape


# ============================================================================
# MODEL LOADING AT STARTUP
# ============================================================================
# Why load at startup instead of per request:
# 1. Loading models is expensive (100-500ms each)
# 2. Models don't change between requests
# 3. Avoids memory leaks from repeated loading
# 4. Enables faster response times (critical for user experience)
# ============================================================================

print("Loading models and configuration...")

try:
    # Load configuration
    # Contains: window_size, latent_dim, required columns, input_dim
    config = joblib.load("models/config.pkl")
    print(f"✓ Config loaded: window_size={config['window_size']}, latent_dim={config['latent_dim']}")
    
    # Load MinMaxScaler
    # Why: Must use EXACT same scaling as training
    scaler = joblib.load("models/scaler.pkl")
    print("✓ Scaler loaded")
    
    # Load encoder model
    # Use the regenerated H5 file
    encoder_model = tf.keras.models.load_model("models/vae_encoder.h5", compile=False)
    print("✓ VAE Encoder loaded")
    
    # Load VAE-LSTM-Attention model
    # Pass SumLayer as custom object
    prediction_model = tf.keras.models.load_model(
        "models/vae_lstm_attention.h5",
        custom_objects={"SumLayer": SumLayer},
        compile=False
    )
    print("✓ VAE-LSTM-Attention model loaded")
    
    print("All models loaded successfully!\n")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    raise RuntimeError(f"Failed to load models at startup: {e}")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="GHG Emission Prediction API",
    description="Inference service for VAE-LSTM-Attention model",
    version="1.0.0"
)

# Enable CORS for frontend communication
# Why: Allows React frontend (different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/")
def health_check():
    """
    Verify that the API is running and models are loaded.
    """
    return {
        "status": "healthy",
        "model": "VAE-LSTM-Attention",
        "window_size": config["window_size"],
        "latent_dim": config["latent_dim"]
    }


# ============================================================================
# PREDICTION ENDPOINT
# ============================================================================

@app.post("/predict")
async def predict_emission(file: UploadFile = File(...)) -> Dict:
    """
    Predict next year's GHG emission from uploaded CSV data.
    
    Process:
    1. Read and validate CSV
    2. Preprocess (sort, clean, extract last N years)
    3. Scale features using saved scaler
    4. Encode to latent space using VAE encoder
    5. Create sequence for LSTM
    6. Generate prediction
    
    Expected CSV format:
    - Must contain columns: year, population, gdp, methane, nitrous_oxide,
      ghg_excluding_lucf_per_capita, energy_per_capita, total_ghg_excluding_lucf
    - Must have at least window_size (5) years of complete data
    - Should be for ONE country only
    
    Returns:
        JSON with predicted emission value
    """
    
    try:
        # ====================================================================
        # STEP 1: Read CSV file
        # ====================================================================
        # Why read as bytes then decode:
        # - UploadFile is async stream, need to read content first
        # - io.StringIO converts bytes to file-like object for pandas
        # ====================================================================
        
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode('utf-8')))
        
        print(f"Received CSV with {len(df)} rows")
        
        
        # ====================================================================
        # STEP 2: Validate CSV structure
        # ====================================================================
        # Why validate:
        # - Missing columns cause cryptic errors later
        # - Better to fail fast with clear error message
        # ====================================================================
        
        required_columns = config["columns_to_keep"]
        is_valid, error_msg = validate_csv_data(df, required_columns)
        
        if not is_valid:
            raise HTTPException(status_code=400, detail=error_msg)
        
        
        # ====================================================================
        # STEP 3: Preprocess data
        # ====================================================================
        # Why preprocess:
        # - Must match EXACT same preprocessing as training
        # - Sort by year (LSTM needs chronological order)
        # - Drop nulls (model trained on complete data)
        # - Extract last N years (we predict year N+1)
        # ====================================================================
        
        df_processed, error_msg = preprocess_input_data(
            df, 
            required_columns, 
            config["window_size"]
        )
        
        if df_processed is None:
            raise HTTPException(status_code=400, detail=error_msg)
        
        print(f"Preprocessed to {len(df_processed)} years (window_size={config['window_size']})")
        
        
        # ====================================================================
        # STEP 4: Scale features
        # ====================================================================
        # Why use saved scaler:
        # - Model expects 0-1 scaled inputs
        # - Must use SAME min/max as training
        # - Using fit_transform would learn NEW min/max (wrong!)
        # ====================================================================
        
        scaled_data = scale_features(df_processed, scaler)
        print(f"Scaled data shape: {scaled_data.shape}")
        
        
        # ====================================================================
        # STEP 5: Encode to latent space
        # ====================================================================
        # Why use VAE encoder:
        # - Final model was trained on latent features, not raw features
        # - VAE reduces 7D -> 3D while preserving relationships
        # - Latent space captures non-linear feature interactions
        # ====================================================================
        
        latent_features = encode_to_latent_space(scaled_data, encoder_model)
        print(f"Latent features shape: {latent_features.shape}")
        
        
        # ====================================================================
        # STEP 6: Create sequence for LSTM
        # ====================================================================
        # Why create sequence:
        # - LSTM processes sequences, not single time points
        # - Need shape (1, window_size, latent_dim) for prediction
        # - 1 = batch size (predicting one future point)
        # ====================================================================
        
        sequence = create_inference_sequence(latent_features, config["window_size"])
        
        # Verify shape matches model expectations
        expected_shape = (1, config["window_size"], config["latent_dim"])
        verify_sequence_shape(sequence, expected_shape)
        
        print(f"Sequence shape: {sequence.shape}")
        
        
        # ====================================================================
        # STEP 7: Generate prediction
        # ====================================================================
        # Why use .predict():
        # - Model outputs scaled prediction (0-1 range)
        # - This represents the latent dimension being predicted
        # - In full pipeline, you'd inverse transform to real emission value
        # ====================================================================
        
        prediction = prediction_model.predict(sequence, verbose=0)
        predicted_value = float(prediction[0][0])  # Extract scalar from array
        
        print(f"Prediction: {predicted_value}")
        
        
        # ====================================================================
        # STEP 8: Return response
        # ====================================================================
        
        return {
            "predicted_value": predicted_value,
            "window_size": config["window_size"],
            "model": "VAE-LSTM-Attention",
            "latent_dim": config["latent_dim"],
            "input_years": len(df_processed),
            "message": "Prediction generated successfully"
        }
    
    
    # ========================================================================
    # ERROR HANDLING
    # ========================================================================
    
    except HTTPException:
        # Re-raise HTTP exceptions (already formatted)
        raise
    
    except Exception as e:
        # Catch unexpected errors and return 500
        print(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error during prediction: {str(e)}"
        )


# ============================================================================
# RUN SERVER
# ============================================================================
# To run: uvicorn app:app --reload --host 0.0.0.0 --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
