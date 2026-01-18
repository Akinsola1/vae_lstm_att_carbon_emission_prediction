"""
Regenerate ALL models and artifacts for TensorFlow 2.16 compatibility.

This script retrains the VAE and LSTM models using the current environment
to ensure all saved files (h5, pkl) are compatible with the installed versions
of TensorFlow (v2.16+), Keras (v3), and NumPy.

Usage:
    python regenerate_models.py
"""

import os
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import keras
from keras import layers, Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

@keras.saving.register_keras_serializable()
class SumLayer(layers.Layer):
    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)
        
    def get_config(self):
        return super().get_config()

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# ============================================================================
# CONFIGURATION
# ============================================================================
WINDOW_SIZE = 5
LATENT_DIM = 3
COLUMNS_TO_KEEP = [
    "year",
    "population",
    "gdp",
    "methane",
    "nitrous_oxide",
    "ghg_excluding_lucf_per_capita",
    "energy_per_capita",
    "total_ghg_excluding_lucf"
]

print(f"TensorFlow Version: {tf.__version__}")
print(f"Keras Version: {keras.__version__}")

# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================
print("\n[1/6] Loading and preprocessing data...")

try:
    df = pd.read_csv("owid-co2-data.csv")
except FileNotFoundError:
    print("Error: 'owid-co2-data.csv' not found in current directory.")
    exit(1)

# Filter for Nigeria
df_ng = df[df["country"] == "Nigeria"].copy()
df_ng = df_ng.sort_values(by="year").reset_index(drop=True)

# Keep required columns
df_ng = df_ng[COLUMNS_TO_KEEP]
df_ng = df_ng.dropna().reset_index(drop=True)

print(f"Data Loaded: {df_ng.shape}")

# Use everything except year for features
train_data = df_ng.drop(columns=["year"])
input_dim = train_data.shape[1]

# ============================================================================
# 2. SCALING
# ============================================================================
print("\n[2/6] Scaling data...")
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(train_data)
df_scaled = pd.DataFrame(scaled_values, columns=train_data.columns)

# Save Scaler
os.makedirs("backend/models", exist_ok=True)
joblib.dump(scaler, "backend/models/scaler.pkl")
print("✓ Scaler saved")

# Save Config
config = {
    "window_size": WINDOW_SIZE,
    "latent_dim": LATENT_DIM,
    "columns_to_keep": COLUMNS_TO_KEEP,
    "input_dim": input_dim
}
joblib.dump(config, "backend/models/config.pkl")
print("✓ Config saved")

# ============================================================================
# 3. VAE MODEL (Keras 3)
# ============================================================================
print("\n[3/6] Building and Training VAE...")

class VAE(keras.Model):
    def __init__(self, input_dim, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        # Encoder
        self.encoder_input = layers.Input(shape=(input_dim,))
        self.encoder_h = layers.Dense(16, activation="relu")
        self.z_mean_layer = layers.Dense(latent_dim)
        self.z_log_var_layer = layers.Dense(latent_dim)
        # Decoder
        self.decoder_h = layers.Dense(16, activation="relu")
        self.decoder_out = layers.Dense(input_dim, activation="sigmoid")
        
    def encode(self, x):
        h = self.encoder_h(x)
        z_mean = self.z_mean_layer(h)
        z_log_var = self.z_log_var_layer(h)
        return z_mean, z_log_var
    
    def reparameterize(self, z_mean, z_log_var):
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    def decode(self, z):
        h = self.decoder_h(z)
        return self.decoder_out(h)
    
    def call(self, inputs):
        z_mean, z_log_var = self.encode(inputs)
        z = self.reparameterize(z_mean, z_log_var)
        reconstructed = self.decode(z)
        # KL Loss custom addition
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        self.add_loss(kl_loss)
        return reconstructed

vae = VAE(input_dim, LATENT_DIM)
vae.compile(optimizer="adam", loss="mse")
vae.fit(scaled_values, scaled_values, epochs=50, batch_size=8, verbose=0)

# Extract Latent Features
z_mean, _ = vae.encode(scaled_values)
latent_features = z_mean.numpy()

# Save Encoder Model
# For Keras 3 saving, we need to construct a functional model for the encoder part
encoder_inputs = keras.Input(shape=(input_dim,))
enc_h = vae.encoder_h(encoder_inputs)
enc_z_mean = vae.z_mean_layer(enc_h)
encoder_model = keras.Model(encoder_inputs, enc_z_mean, name="encoder")

# Use .keras extension for Keras 3 native format, but we'll stick to .h5 if app expects it.
# However, Keras 3 .h5 might still be problematic with legacy loading. 
# We will save as .h5 but it will be a "new" .h5.
encoder_model.save("backend/models/vae_encoder.h5")
print("✓ VAE Encoder saved")

# ============================================================================
# 4. SEQUENCE CREATION
# ============================================================================
print("\n[4/6] Creating Sequences...")

def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size, 0]) # Target: 1st latent dim (approx) or total_ghg?
        # Note: Original notebook used 'total_ghg' column index for target.
        # But here we are using LATENT features.
        # In Turn 5, user used: target_col=0 (First latent dim)
    return np.array(X), np.array(y)

# Latent features are typically used for inputs. 
# The target MUST be consistent with what we want to predict.
# If we predict latent, we need a decoder in backend.
# The original code provided by user suggests predicting 'total_ghg_excluding_lucf'.
# BUT in Turn 6, user specificied: target_col=0 (First latent dimension).
# So the model predicts the first latent dimension. 
# (This implies the backend result needs interpretation, but we follow the notebook logic).

X_seq, y_seq = create_sequences(latent_features, WINDOW_SIZE)
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# ============================================================================
# 5. LSTM-ATTENTION MODEL (Keras 3)
# ============================================================================
print("\n[5/6] Building and Training LSTM-Attention...")

# Functional API for Attention
inputs = keras.Input(shape=(WINDOW_SIZE, LATENT_DIM))
lstm_out = layers.LSTM(32, return_sequences=True)(inputs)

# Attention Block
# Keras 3 ops
att_weights = layers.Dense(1, activation="tanh")(lstm_out)
att_weights = layers.Softmax(axis=1)(att_weights)
attended = layers.Multiply()([lstm_out, att_weights])
# Lambda using tf ops
# Custom SumLayer
context = SumLayer()(attended)

output = layers.Dense(1)(context)

model = keras.Model(inputs, output)
model.compile(optimizer="adam", loss="mse")
model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

model.save("backend/models/vae_lstm_attention.h5")
print("✓ LSTM-Attention Model saved")

print("\n[6/6] All artifacts regenerated successfully!")
