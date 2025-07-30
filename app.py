import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# --- Your windowing function ---
def df_to_windowed_df(dataframe, firstdatestr, lastdatestr, n=3):
    first_date = pd.to_datetime(firstdatestr)
    last_date = pd.to_datetime(lastdatestr)
    dates, X, Y = [], [], []

    dataframe = dataframe.sort_index()
    if not isinstance(dataframe.index, pd.DatetimeIndex):
        dataframe.index = pd.to_datetime(dataframe.index)

    all_dates = dataframe.loc[first_date:last_date].index

    for target_date in all_dates:
        df_subset = dataframe.loc[:target_date].tail(n+1)
        if len(df_subset) != n+1:
            continue
        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]
        dates.append(target_date)
        X.append(x)
        Y.append(y)

    ret_df = pd.DataFrame({'Date': dates})
    X = np.array(X)
    for i in range(n):
        ret_df[f'Target-{n-i}'] = X[:, i]
    ret_df['Target'] = Y
    return ret_df

st.title("Stock Data Polishing & Prediction Tool")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Raw Data Preview:", df.head())

    # Ensure Date column is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    else:
        st.error("No 'Date' column found in uploaded file.")
        st.stop()

    # Ask for date range
    min_date = df.index.min().date()
    max_date = df.index.max().date()
    st.write(f"Date range in data: {min_date} to {max_date}")

    start_date = st.date_input("Start Date", min_value=min_date, max_value=max_date, value=min_date)
    end_date = st.date_input("End Date", min_value=min_date, max_value=max_date, value=max_date)
    n = st.number_input("Window Size (N)", min_value=1, max_value=30, value=3)

    if st.button("Data Polish"):
        # Plot Open and Close
        fig, ax = plt.subplots()
        df[['Open', 'Close']].plot(ax=ax)
        st.pyplot(fig)

        # Run windowing
        windowed_df = df_to_windowed_df(df, str(start_date), str(end_date), n)
        st.write("Windowed Data:", windowed_df)

        # Save to session state
        st.session_state['windowed_df'] = windowed_df
        st.session_state['df'] = df
        st.session_state['n'] = n

        # Download windowed data
        csv = windowed_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Windowed Data CSV", csv, "windowed_data.csv", "text/csv")

# Prediction section (always visible if windowed_df exists)
if 'windowed_df' in st.session_state:
    st.subheader("Prediction")
    pred_days = st.number_input("How many future days to predict?", min_value=1, max_value=30, value=5)
    if st.button("Create Prediction"):
        windowed_df = st.session_state['windowed_df']
        df = st.session_state['df']
        n = st.session_state['n']
        # Train a simple model (Linear Regression)
        X_train = windowed_df[[f'Target-{i+1}' for i in range(n)]].values
        y_train = windowed_df['Target'].values
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Start with the last N closes from the original df
        last_known = df['Close'].iloc[-n:].values.tolist()
        future_preds = []
        future_dates = []
        last_date = df.index[-1]
        for i in range(pred_days):
            next_pred = model.predict([last_known[-n:]])[0]
            future_preds.append(next_pred)
            # Generate next date (assume business days)
            next_date = pd.bdate_range(last_date, periods=2)[-1]
            future_dates.append(next_date)
            # Update for next prediction
            last_known.append(next_pred)
            last_date = next_date

        pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds})
        st.write("Predicted Next N Days:", pred_df)

        # Plot predictions
        fig2, ax2 = plt.subplots()
        df['Close'].plot(ax=ax2, label='Historical Close')
        pred_df.set_index('Date')['Predicted Close'].plot(ax=ax2, style='--o', color='red', label='Predicted')
        ax2.legend()
        st.pyplot(fig2)