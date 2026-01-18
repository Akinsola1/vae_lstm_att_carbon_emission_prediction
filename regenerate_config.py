"""
Regenerate config.pkl file for backend inference.

Run this if config.pkl is corrupted or missing.
This must be run from the project root directory.
"""

import joblib

# Configuration matching the training setup
config = {
    "window_size": 5,  # Number of historical years needed for prediction
    "latent_dim": 3,   # VAE latent space dimensions
    "columns_to_keep": [
        "year",
        "population",
        "gdp",
        "methane",
        "nitrous_oxide",
        "ghg_excluding_lucf_per_capita",
        "energy_per_capita",
        "total_ghg_excluding_lucf"
    ],
    "input_dim": 7  # Number of features (excluding year)
}

# Save configuration
joblib.dump(config, "backend/models/config.pkl")
print("✓ Configuration saved successfully to backend/models/config.pkl")
print(f"  - window_size: {config['window_size']}")
print(f"  - latent_dim: {config['latent_dim']}")
print(f"  - input_dim: {config['input_dim']}")
print(f"  - columns: {len(config['columns_to_keep'])} features")
