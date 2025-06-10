"""
Creates a test XGBoost model for predicting short-term price movements.
"""
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split

# --- Configuration ---
MODEL_DIR = Path('models/')
MODEL_NAME = 'xgboost_model.joblib'
MODEL_PATH = MODEL_DIR / MODEL_NAME
LOOKAHEAD_PERIOD = 6 # Corresponds to 30 minutes with 5m candles

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# --- 1. Generate Synthetic Feature Data ---
print("Generating synthetic multi-timeframe feature data...")
num_samples = 2000
# Mimic output of FeatureExtractor
data = {
    'close': np.random.uniform(100, 110, size=num_samples),
    'sma_short_15m': np.random.uniform(98, 112, size=num_samples),
    'rsi_15m': np.random.uniform(30, 70, size=num_samples),
    'sma_long_1h': np.random.uniform(95, 115, size=num_samples),
    'rsi_1h': np.random.uniform(20, 80, size=num_samples),
}
X = pd.DataFrame(data)

# --- 2. Generate Synthetic Target Variable ---
# The target is: will the price be higher in `LOOKAHEAD_PERIOD` candles?
# We shift the 'close' price backwards to get the future price at each point.
future_price = X['close'].shift(-LOOKAHEAD_PERIOD)
y = (future_price > X['close']).astype(int)

# Remove the last rows where the future price is unknown (NaN)
X = X.iloc[:-LOOKAHEAD_PERIOD]
y = y.iloc[:-LOOKAHEAD_PERIOD]

print(f"Generated {len(y)} samples, with {y.sum()} positive examples.")

# --- 3. Train the XGBoost Model ---
print("Training an XGBoost Classifier model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3
)
model.fit(X_train, y_train)

# --- 4. Evaluate and Save the Model ---
accuracy = model.score(X_test, y_test)
print(f"Model accuracy on test set: {accuracy:.2f}")

print(f"Saving model to {MODEL_PATH}...")
joblib.dump(model, MODEL_PATH)

print("\nXGBoost test model created and saved successfully.")
print(f"This model expects a DataFrame with columns: {list(X.columns)}") 