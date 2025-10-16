import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import joblib
import datetime

# ---------------------------
# 1. Create Synthetic Market Data
# ---------------------------
np.random.seed(42)

dates = pd.date_range(start="2023-01-01", periods=300)
prices = np.cumsum(np.random.randn(300)) + 100  # Random walk around 100

df = pd.DataFrame({
    "date": dates,
    "close": prices
})

df.to_csv("synthetic_market.csv", index=False)
print("✅ synthetic_market.csv saved successfully!")

# ---------------------------
# 2. Preprocess for LSTM
# ---------------------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_prices = scaler.fit_transform(df["close"].values.reshape(-1, 1))

joblib.dump(scaler, "lstm_scaler.pkl")
print("✅ lstm_scaler.pkl saved successfully!")

n_steps = 10
X, y = [], []

for i in range(n_steps, len(scaled_prices)):
    X.append(scaled_prices[i - n_steps:i, 0])
    y.append(scaled_prices[i, 0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# ---------------------------
# 3. Build Simple LSTM Model
# ---------------------------
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(n_steps, 1)),
    LSTM(50, return_sequences=False),
    Dense(1)
])

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(X, y, epochs=10, batch_size=16, verbose=1)

# ---------------------------
# 4. Save Model
# ---------------------------
model.save("lstm_close_model.h5")
print("✅ lstm_close_model.h5 saved successfully!")

print("\n🎉 Training complete! You can now run your FastAPI app.")
