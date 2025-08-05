import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

# --- Feature Engineering ---
def add_features(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['LogRet'] = np.log(df['Close'] / df['Close'].shift(1))
    df = df.dropna()
    return df

# --- Streamlit UI ---
st.title("LSTM + XGBoost: Close Price Prediction")

uploaded_file = st.file_uploader("Upload your stock CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'Date' not in df.columns:
        st.error("CSV must have a 'Date' column.")
        st.stop()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = add_features(df)

    if st.button("🧹 Data Polish"):
        st.write("### Preview after Feature Engineering:")
        st.dataframe(df.tail(10))
        st.line_chart(df[['Close', 'MA10', 'MA20']])
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Volume'], label='Volume', color='gray')
        ax.set_title("Volume Over Time")
        ax.legend()
        st.pyplot(fig)
        st.success("Feature Engineering applied successfully.")

    min_date, max_date = df.index.min().date(), df.index.max().date()
    st.write(f"Date range: {min_date} to {max_date}")

    n = st.number_input("Window Size (N)", min_value=5, max_value=50, value=20)

    if st.button("Train and Predict Next 2 Days Close"):
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'Volatility', 'LogRet']
        df = df.dropna()

        # Prepare input data
        X_all = []
        Y_all = []
        for i in range(n, len(df) - 2):
            window = df[features].iloc[i - n:i].values
            target = df['Close'].iloc[i:i+2].values  # next 2 days close prices
            if len(target) == 2:
                X_all.append(window)
                Y_all.append(target)

        X_all = np.array(X_all)
        Y_all = np.array(Y_all)

        # Scale features
        x_scalers = [MinMaxScaler() for _ in features]
        X_scaled = np.zeros_like(X_all)
        for i, scaler in enumerate(x_scalers):
            X_feat = X_all[:, :, i].reshape(-1, 1)
            X_scaled[:, :, i] = scaler.fit_transform(X_feat).reshape(-1, n)

        y_scaler = MinMaxScaler()
        Y_scaled = y_scaler.fit_transform(Y_all)

        split = int(0.8 * len(X_scaled))
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        Y_train, Y_test = Y_scaled[:split], Y_scaled[split:]

        # Build LSTM model
        model = Sequential([
            Input(shape=(n, len(features))),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(2)  # Predict next 2 days of Close only
        ])
        model.compile(optimizer='adam', loss=Huber())
        model.fit(X_train, Y_train, epochs=50, batch_size=16, validation_split=0.2, callbacks=[EarlyStopping(patience=5, restore_best_weights=True)], verbose=0)

        # Evaluate LSTM
        Y_pred = model.predict(X_test)
        Y_pred_inv = y_scaler.inverse_transform(Y_pred)
        Y_test_inv = y_scaler.inverse_transform(Y_test)

        mae = mean_absolute_error(Y_test_inv, Y_pred_inv)
        st.write(f"\U0001F4CA LSTM MAE on 2-day Close Prediction: {mae:.4f}")

        fig, ax = plt.subplots()
        ax.plot(Y_test_inv[:, 0], label='True Day+1')
        ax.plot(Y_pred_inv[:, 0], label='Pred Day+1')
        ax.plot(Y_test_inv[:, 1], label='True Day+2', linestyle='--')
        ax.plot(Y_pred_inv[:, 1], label='Pred Day+2', linestyle='--')
        ax.set_title("LSTM Close Price Predictions")
        ax.legend()
        st.pyplot(fig)

        # XGBoost Benchmark (flattened window)
        X_flat = X_scaled.reshape(X_scaled.shape[0], -1)
        X_train_flat, X_test_flat = X_flat[:split], X_flat[split:]
        st.markdown("### \U0001F50D XGBoost Baseline")
        for i in range(2):
            model_xgb = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
            model_xgb.fit(X_train_flat, Y_train[:, i])
            pred_xgb = model_xgb.predict(X_test_flat)
            pred_xgb_inv = y_scaler.inverse_transform(np.column_stack([pred_xgb if j==i else np.zeros_like(pred_xgb) for j in range(2)]))[:, i]
            true_inv = Y_test_inv[:, i]
            mae_xgb = mean_absolute_error(true_inv, pred_xgb_inv)
            st.write(f"XGBoost MAE for Day+{i+1}: {mae_xgb:.4f}")

        # Predict future (last available window)
        last_window = df[features].iloc[-n:].values
        last_scaled = np.zeros((1, n, len(features)))
        for i, scaler in enumerate(x_scalers):
            last_scaled[0, :, i] = scaler.transform(last_window[:, i].reshape(-1, 1)).flatten()
        future_pred = model.predict(last_scaled)
        future_close = y_scaler.inverse_transform(future_pred)[0]

        next_days = pd.bdate_range(df.index[-1], periods=3)[1:]
        st.write("\U0001F4C8 LSTM Predicted Close for next 2 business days:")
        st.write({str(next_days[0].date()): future_close[0], str(next_days[1].date()): future_close[1]})
