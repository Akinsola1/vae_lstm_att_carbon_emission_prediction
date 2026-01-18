"""
Regenerate scaler.pkl with current numpy version.

This must be run from the Jupyter notebook or you need to have the training data available.
Run this AFTER upgrading numpy to fix pickle compatibility.
"""

import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

print("Loading data...")
df = pd.read_csv("owid-co2-data.csv")
df_ng = df[df["country"] == "Nigeria"]
df_ng = df_ng.sort_values(by="year").reset_index(drop=True)

columns_to_keep = [
    "year",
    "population",
    "gdp",
    "methane",
    "nitrous_oxide",
    "ghg_excluding_lucf_per_capita",
    "energy_per_capita",
    "total_ghg_excluding_lucf"
]

df_ng = df_ng[columns_to_keep]
df_ng = df_ng.dropna().reset_index(drop=True)

print(f"Data shape: {df_ng.shape}")

# Create and fit scaler
print("Creating scaler...")
scaler = MinMaxScaler()
scaler.fit(df_ng.drop(columns=["year"]).values)

# Save scaler
joblib.dump(scaler, "backend/models/scaler.pkl")
print("✓ Scaler saved successfully to backend/models/scaler.pkl")
print(f"  - Min values: {scaler.data_min_[:3]}...")
print(f"  - Max values: {scaler.data_max_[:3]}...")
