import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.losses import Huber
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

# --- Enhanced Feature Engineering ---
def add_features(df):
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=10).std()
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    for lag in range(1, 6):
        df[f'Lag_{lag}'] = df['Close'].shift(lag)
    return df.dropna()

# --- LSTM Model Definition ---
def build_lstm_model(n_timesteps, n_features, out_steps):
    model = Sequential([
        Input(shape=(n_timesteps, n_features)),
        LSTM(128, return_sequences=True),
        Dropout(0.2),
        LSTM(64),
        Dense(32, activation='relu'),
        Dense(out_steps)
    ])
    model.compile(optimizer='adam', loss=Huber())
    return model

# --- Streamlit App ---
st.title("\U0001F4C8 LSTM + XGBoost: Predict Future Close Prices")

model_choice = st.selectbox("Select Model", ["LSTM", "XGBoost", "Both"])
predict_days = st.slider("Predict N Future Days", min_value=1, max_value=10, value=2)

uploaded_file = st.file_uploader("Upload your stock CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if 'Date' not in df.columns:
        st.error("CSV must have a 'Date' column.")
        st.stop()

    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date').sort_index()
    df = add_features(df)

    st.write(f"Date Range: **{df.index.min().date()}** to **{df.index.max().date()}**")
    n = st.number_input("\U0001F4CF Window Size (N)", min_value=5, max_value=50, value=20)

    if st.button("\U0001F9F9 Data Polish"):
        st.subheader("Feature Engineered Preview")
        st.dataframe(df.tail(10))
        st.line_chart(df[['Close', 'MA10', 'MA20']])
        fig, ax = plt.subplots()
        ax.plot(df.index, df['Volume'], color='gray')
        ax.set_title("Volume Over Time")
        st.pyplot(fig)

    if st.button("\U0001F680 Train and Predict"):
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA20', 'Volatility', 'Momentum'] + [f'Lag_{i}' for i in range(1, 6)]
        df = df.dropna()

        for i in range(1, predict_days + 1):
            df[f'Close_Day+{i}'] = df['Close'].shift(-i)
        df.dropna(inplace=True)

        X = df[features]
        Y = df[[f'Close_Day+{i}' for i in range(1, predict_days + 1)]]

        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X)
        scaler_Y = MinMaxScaler()
        Y_scaled = scaler_Y.fit_transform(Y)

        split = int(0.8 * len(X))
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        Y_train, Y_test = Y_scaled[:split], Y_scaled[split:]

        if model_choice in ["LSTM", "Both"]:
            st.subheader("\U0001F52C LSTM Evaluation")
            X_lstm = []
            Y_lstm = []
            for i in range(n, len(X_scaled)):
                X_lstm.append(X_scaled[i - n:i])
                Y_lstm.append(Y_scaled[i])
            X_lstm = np.array(X_lstm)
            Y_lstm = np.array(Y_lstm)

            split_lstm = int(0.8 * len(X_lstm))
            X_lstm_train, X_lstm_test = X_lstm[:split_lstm], X_lstm[split_lstm:]
            Y_lstm_train, Y_lstm_test = Y_lstm[:split_lstm], Y_lstm[split_lstm:]

            model_lstm = build_lstm_model(n, X_lstm.shape[2], predict_days)
            es = EarlyStopping(patience=10, restore_best_weights=True)
            lr_plateau = ReduceLROnPlateau(patience=5)

            model_lstm.fit(X_lstm_train, Y_lstm_train, epochs=50, batch_size=16,
                           validation_split=0.2, callbacks=[es, lr_plateau], verbose=0)

            pred_lstm = model_lstm.predict(X_lstm_test)
            Y_test_inv = scaler_Y.inverse_transform(Y_lstm_test)
            Y_pred_inv = scaler_Y.inverse_transform(pred_lstm)

            for i in range(predict_days):
                mae = mean_absolute_error(Y_test_inv[:, i], Y_pred_inv[:, i])
                rmse = np.sqrt(mean_squared_error(Y_test_inv[:, i], Y_pred_inv[:, i]))
                st.write(f"**LSTM MAE for Day+{i+1}:** `{mae:.4f}`, **RMSE:** `{rmse:.4f}`")

                fig, ax = plt.subplots()
                ax.plot(Y_test_inv[:, i], label='True')
                ax.plot(Y_pred_inv[:, i], label='LSTM Pred')
                ax.set_title(f"LSTM Prediction - Day+{i+1}")
                ax.legend()
                st.pyplot(fig)

            future_pred = model_lstm.predict(X_lstm[-1].reshape(1, n, X_lstm.shape[2]))
            future_inv = scaler_Y.inverse_transform(future_pred).flatten()

            st.subheader("\U0001F4C5 LSTM Predicted Future Close Prices")
            future_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=predict_days)
            future_forecast = {str(future_dates[i].date()): float(future_inv[i]) for i in range(predict_days)}
            st.write(future_forecast)

        if model_choice in ["XGBoost", "Both"]:
            st.subheader("\U0001F50D XGBoost Improved")
            for i in range(predict_days):
                model_xgb = xgb.XGBRegressor(
                    objective='reg:squarederror',
                    n_estimators=300,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8
                )
                model_xgb.fit(X_train, Y_train[:, i])
                pred_xgb = model_xgb.predict(X_test)
                mae = mean_absolute_error(Y_test[:, i], pred_xgb)
                rmse = np.sqrt(mean_squared_error(Y_test[:, i], pred_xgb))

                st.write(f"**XGBoost MAE for Day+{i+1}:** `{mae:.4f}`, **RMSE:** `{rmse:.4f}`")

                dummy = np.zeros_like(Y_test)
                dummy[:, i] = pred_xgb
                pred_inv = scaler_Y.inverse_transform(dummy)[:, i]
                true_inv = scaler_Y.inverse_transform(Y_test)[:, i]

                fig, ax = plt.subplots()
                ax.plot(true_inv, label='True')
                ax.plot(pred_inv, label='XGBoost Pred')
                ax.set_title(f"XGBoost Prediction - Day+{i+1}")
                ax.legend()
                st.pyplot(fig)

                fig2, ax2 = plt.subplots()
                xgb.plot_importance(model_xgb, ax=ax2)
                ax2.set_title(f"Feature Importance - Day+{i+1}")
                st.pyplot(fig2)
