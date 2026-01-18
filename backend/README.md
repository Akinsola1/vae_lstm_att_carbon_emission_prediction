# Backend Inference Service

## Overview

This is an **inference-only** FastAPI service for the VAE-LSTM-Attention GHG emission prediction model.

**Important**: This backend does NOT train models. All models are pre-trained and loaded at startup.

## Architecture

```
backend/
├── app.py                          # FastAPI application & endpoints
├── models/                         # Pre-trained model artifacts
│   ├── vae_encoder.h5             # VAE encoder (7D -> 3D)
│   ├── vae_lstm_attention.h5      # Final prediction model
│   ├── scaler.pkl                 # MinMaxScaler from training
│   └── config.pkl                 # Training configuration
├── utils/
│   ├── preprocess.py              # Data preprocessing utilities
│   └── sequence.py                # Sequence creation for LSTM
└── requirements.txt               # Python dependencies
```

## Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Verify Models Exist

Ensure these files exist in `backend/models/`:
- `vae_encoder.h5`
- `vae_lstm_attention.h5`
- `scaler.pkl`
- `config.pkl`

If missing, run the training notebook (`setup.ipynb`) first.

### 3. Run Server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

Server will start at: `http://localhost:8000`

## API Endpoints

### Health Check

```bash
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "model": "VAE-LSTM-Attention",
  "window_size": 5,
  "latent_dim": 3
}
```

### Prediction

```bash
POST /predict
```

**Request:**
- Content-Type: `multipart/form-data`
- Body: CSV file with required columns

**Required CSV Columns:**
- `year`
- `population`
- `gdp`
- `methane`
- `nitrous_oxide`
- `ghg_excluding_lucf_per_capita`
- `energy_per_capita`
- `total_ghg_excluding_lucf`

**Minimum Data:** At least 5 years of complete (no missing values) data

**Response:**
```json
{
  "predicted_value": 0.456,
  "window_size": 5,
  "model": "VAE-LSTM-Attention",
  "latent_dim": 3,
  "input_years": 5,
  "message": "Prediction generated successfully"
}
```

## Testing

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_data.csv"
```

### Using Python

```python
import requests

with open("test_data.csv", "rb") as f:
    files = {"file": f}
    response = requests.post("http://localhost:8000/predict", files=files)
    print(response.json())
```

## How It Works

### 1. Model Loading (Startup)
- All models loaded **once** at startup
- Avoids expensive reload on each request
- Ensures consistent state

### 2. Data Preprocessing Pipeline
1. **Validation**: Check required columns exist
2. **Sorting**: Order by year (time series requirement)
3. **Cleaning**: Drop rows with missing values
4. **Windowing**: Extract last N years (N = window_size)

### 3. Feature Transformation
1. **Scaling**: Apply saved MinMaxScaler (0-1 range)
2. **VAE Encoding**: Transform 7D features → 3D latent space
3. **Sequencing**: Create LSTM input shape (1, window_size, latent_dim)

### 4. Prediction
- Run model.predict() on sequence
- Return predicted value for next year

## Why Each Step Matters

### Why Load Models at Startup?
- Loading models takes 100-500ms
- Models are static (don't change between requests)
- Faster response times for users

### Why Use Saved Scaler?
- Model trained on 0-1 scaled data
- Using different scale = wrong predictions
- Must use EXACT min/max from training

### Why VAE Encoding?
- Reduces dimensionality (7 → 3)
- Captures non-linear relationships
- Model was trained on latent features

### Why Sequence Shape Matters?
- LSTM expects (batch, timesteps, features)
- Shape mismatch = immediate failure
- Must match training exactly

## Error Handling

Common errors and solutions:

### Missing Columns
```json
{
  "detail": "Missing required columns: {'gdp', 'methane'}"
}
```
**Solution**: Ensure CSV has all required columns

### Insufficient Data
```json
{
  "detail": "Need at least 5 years of data, got 3"
}
```
**Solution**: Provide at least window_size years

### Shape Mismatch
```json
{
  "detail": "Sequence shape mismatch. Expected (1, 5, 3), got (1, 4, 3)"
}
```
**Solution**: Check preprocessing logic

## Production Considerations

This is a research-grade backend. For production:

1. **Add authentication** (API keys, OAuth)
2. **Rate limiting** (prevent abuse)
3. **Logging** (structured logs for debugging)
4. **Monitoring** (track prediction latency)
5. **Input validation** (Pydantic models)
6. **CORS** (restrict to specific frontend URL)
7. **HTTPS** (TLS encryption)
8. **Containerization** (Docker)

## Dependencies

See `requirements.txt` for full list.

Key dependencies:
- **FastAPI**: Web framework
- **TensorFlow**: Model inference
- **Pandas**: Data processing
- **Scikit-learn**: Scaler utilities
- **Joblib**: Model serialization

## License

Research use only.
