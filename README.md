
---

## 📊 Data

All models use the file:  
`Data_for_Analysis/PLTR.csv`  
which contains Open, High, Low, Close, and Volume data with dates.

---

## ⚙️ Core Scripts

### 🧪 `DataPolish.ipynb`

- Cleans and visualizes the original dataset
- Converts `Date` into an index
- Uses matplotlib for plotting `Open`, `Close`, `High`, `Low` movements
- Defines and tests multiple **windowing functions**:
  - `df_to_windowed_df()` – for single-feature (Close) windowing
  - `df_to_windowed_df_multi()` – for multiple OHLC features
  - `df_to_windowed_DL()` – tailored to supervised learning for Deep Learning models



THE LOGIC BEHIND THE SCRIPT here is 

## 🧼 Data Polishing and Windowing (for Deep Learning Forecasting)

This section describes the `df_to_windowed_DL(...)` function and how it transforms your time-series stock data into a format suitable for supervised learning, particularly for training models like RNNs, LSTMs, or other deep learning architectures.

---

### 1. 📅 Data Input and Preprocessing

**Expected Input Format:**\
The input is a DataFrame with columns:

```
Date, Open, High, Low, Close
```

This can come from a CSV file or be passed in via code.

**Initial Steps:**

- Sort the DataFrame by date.
- Set `Date` as the index using:

```python
dataframe.index = dataframe.pop('Date')
```

- Make sure the index is in datetime format:

```python
if not isinstance(dataframe.index, pd.DatetimeIndex):
    dataframe.index = pd.to_datetime(dataframe.index)
```

This ensures proper time-based slicing. The `isinstance(...)` check prevents unnecessary conversion.

**Date Selection (Optional):**\
In the Streamlit app, the user can select a start date, end date, and the number of past days (`n`) to use for training. These values are passed into the function.

---

### 2. 🪟 What is Data Windowing?

Windowing means:

> Using `n` previous days of data to predict the next day.

We define the features used:

```python
features = ['Open', 'High', 'Low', 'Close']
```

- **Inputs (X):** Previous `n` days of feature values.
- **Output (Y):** Values of all features on the next day.

**Terminology:**

- `n`: Number of past days (default = 3)
- `F`: Number of features (here, 4)
- `S`: Number of samples (varies with dataset size)

Each sample `xi` is a `(n, F)` matrix:

```python
xi = [
  [o1, h1, l1, c1],
  [o2, h2, l2, c2],
  [o3, h3, l3, c3]
]
```

Its corresponding target `yi` is:

```python
yi = [o4, h4, l4, c4]
```

---

### 3. 🔄 How the Windowing Logic Works

Loop through each date in the range:

```python
for target_date in all_dates:
    df_subset = dataframe.loc[:target_date].tail(n + 1)
    if len(df_subset) != n + 1:
        continue
```

This selects the last `n` rows + 1 target row.

Split into inputs and outputs:

```python
x = df_subset[features].iloc[:-1].to_numpy()  # shape = (n, F)
y = df_subset[features].iloc[-1].to_numpy()   # shape = (F,)
```

- `x[i][0]`: values from 3 days before
- `x[i][1]`: values from 2 days before
- `x[i][2]`: values from 1 day before

Append:

```python
X.append(x)
Y.append(y)
```

Then convert to numpy arrays:

```python
X = np.array(X)  # shape = (S, n, F)
Y = np.array(Y)  # shape = (S, F)
```

---

### 4. 📾 Building the Final Output DataFrame

Start with:

```python
ret_df = pd.DataFrame({'Date': dates})
```

Now flatten the 3D array X into tabular columns:

```python
for time_step in range(n):
    for feat_idx, feat_name in enumerate(features):
        ret_df[f'{feat_name}@Target-{n - time_step}'] = X[:, time_step, feat_idx]
```

Example columns:

- `Open@Target-3`
- `High@Target-2`
- `Low@Target-1`

Now add the real target values:

```python
for feat_idx, feat_name in enumerate(features):
    ret_df[f'{feat_name}@Target'] = Y[:, feat_idx]
```

This gives:

- `Open@Target`, `High@Target`, etc.

---

### ✅ Final Output Example (n = 3)

| Date       | Open\@Target-3 | High\@Target-2 | Low\@Target-1 | ... | Open\@Target | High\@Target | ... |
| ---------- | -------------- | -------------- | ------------- | --- | ------------ | ------------ | --- |
| 2024-07-04 | 10             | 15             | 13            | ... | 13           | 18           |     |
| 2024-07-05 | 11             | 16             | 14            | ... | 14           | 19           |     |
| ...        | ...            | ...            | ...           | ... | ...          | ...          |     |

Each row = one training sample:

- Input: previous `n` days of OHLC
- Target: values of OHLC on the next day

The final `ret_df` is clean, structured, and ready to be fed into an LSTM or any supervised learning model.


## 🚀 Streamlit Apps

### `app.py` – **Basic Stock Windowing & Linear Prediction**

- Upload a CSV and specify date ranges and window size
- Applies windowing to predict next `Close` price using **Linear Regression**
- Visualizes original and predicted data
- Allows download of the windowed dataset


**Why it matters:**  
Introduces the concept of time-windowed prediction, useful as a stepping stone to more advanced ML models.

---

### `app2.py` – **LSTM + XGBoost for Short-Term Prediction**

- Performs advanced feature engineering:
  - Moving Averages (MA10, MA20)
  - Volatility
  - Log Returns
- Builds:
  - **LSTM** model to predict next 2 `Close` prices
  - **XGBoost** benchmark model for comparison
- Visualizes both predictions and error metrics

**What’s different:**
- Combines both deep learning and tree-based methods
- Introduces feature scaling and multi-output regression
- Focuses on short-term (2-day) forecasting

---

### `app3.py` – **Multi-Day Forecasting with LSTM and XGBoost**

- Further improves feature engineering:
  - Adds momentum and lag-based features
- Supports **N-day forecasting** (selectable by user)
- Allows selection of:
  - `LSTM`
  - `XGBoost`
  - or **Both** models
- Displays:
  - MAE/RMSE for each day
  - Feature importance (for XGBoost)
  - Forecasted future prices

**What’s new here:**
- Multi-step forecasting (not just one-shot)
- LSTM with 2 stacked layers
- Use of `ReduceLROnPlateau` for learning rate tuning

---

## 🧠 Learning Outcomes

This project is ideal for:
- Understanding the **differences between traditional and deep learning** forecasting
- Learning how to engineer time-series features
- Practicing **model evaluation metrics** for multi-output regression
- Visualizing financial predictions with minimal code overhead

---

## 🛠 Setup

```bash
# Create and activate environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run any app
streamlit run app.py
streamlit run app2.py
streamlit run app3.py
