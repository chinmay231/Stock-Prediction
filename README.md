
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



THE LOGIC BEHIND THE SCRIPT here is - Here’s a refined and structured version of your explanation, written in clear and professional documentation style—suitable for a README.md or inline code documentation on GitHub. The original content has been preserved in meaning but organized for clarity and precision:

🧼 Data Polishing and Windowing Explanation (for Deep Learning Forecasting)
This section describes the df_to_windowed_DL(...) function and how it transforms your time-series stock data into a format suitable for supervised learning, particularly for training models like RNNs, LSTMs, or other deep learning architectures.

1. Data Input and Preprocessing
Expected Input Format: The input is a DataFrame with columns:
Date, Open, High, Low, Close — either from a CSV or directly injected via code.

Initial Steps:

Sort the DataFrame by Date.

Set Date as the DataFrame index using dataframe.index = dataframe.pop('Date').

Ensure the index is of datetime type:

python
Copy
Edit
if not isinstance(dataframe.index, pd.DatetimeIndex):
    dataframe.index = pd.to_datetime(dataframe.index)
This ensures proper time-based slicing later on. The isinstance(...) check confirms whether the index is already in datetime format.

Date Selection (Optional):
If integrated into a Streamlit interface, users can select a start date, end date, and window size n dynamically using a slider or date input. These values are then passed into the windowing function.

2. What is Data Windowing?
Data windowing transforms sequential time-series data into input-output samples suitable for supervised learning.

Features Used:

python
Copy
Edit
features = ['Open', 'High', 'Low', 'Close']
Inputs: The previous n days of feature values.

Output (target): The value of all features on the target date (the day following the n inputs).

Terminology:

n: Number of previous time steps (default: 3).

F: Number of features (here, 4).

S: Number of samples/windows (derived from available data and n).

Each training sample xi is a matrix of shape (n, F):

python
Copy
Edit
xi = [
  [o1, h1, l1, c1],
  [o2, h2, l2, c2],
  [o3, h3, l3, c3]
]
Its corresponding target yi is:

python
Copy
Edit
yi = [o4, h4, l4, c4]
3. How the Windowing Logic Works
The following logic is executed:

Step-by-step:
Loop through each date in the selected range:

python
Copy
Edit
for target_date in all_dates:
    df_subset = dataframe.loc[:target_date].tail(n + 1)
    if len(df_subset) != n + 1:
        continue  # skip if not enough data
Split into input (X) and output (Y):

python
Copy
Edit
x = df_subset[features].iloc[:-1].to_numpy()  # Shape: (n, F)
y = df_subset[features].iloc[-1].to_numpy()   # Shape: (F,)
Aggregate:

Each x becomes a sample xi added to X

Each y becomes a target yi added to Y

Convert lists into numpy arrays:

python
Copy
Edit
X = np.array(X)  # Shape: (S, n, F)
Y = np.array(Y)  # Shape: (S, F)
4. Building the Final Output DataFrame (ret_df)
We now convert the 3D array X into a 2D DataFrame suitable for ML training. Each column represents a specific feature at a specific timestep in the window.

Structure:
X has shape (S, n, F)

Each X[i] is a matrix of shape (n, F)

Example: if n=3 and F=4, then X[i] is:

python
Copy
Edit
[
  [Open@t-3, High@t-3, Low@t-3, Close@t-3],
  [Open@t-2, High@t-2, Low@t-2, Close@t-2],
  [Open@t-1, High@t-1, Low@t-1, Close@t-1]
]
Column Creation:
python
Copy
Edit
for time_step in range(n):
    for feat_idx, feat_name in enumerate(features):
        ret_df[f'{feat_name}@Target-{n - time_step}'] = X[:, time_step, feat_idx]
Example:
X[:, 0, 0] → values of 'Open' 3 days before target date

X[:, 1, 1] → values of 'High' 2 days before target date

...

X[:, 2, 3] → 'Close' 1 day before target

This gives one column per (time_step, feature) pair. So if n=3 and F=4, you'll have n * F = 12 input columns.

Add Target (Y) Columns:
python
Copy
Edit
for feat_idx, feat_name in enumerate(features):
    ret_df[f'{feat_name}@Target'] = Y[:, feat_idx]
This simply maps the final prediction value (the true value at time t) for each feature.

5. Summary
Each training example is:

X[i]: (n, F) → input window of past n days

Y[i]: (F,) → target values at the target date

We convert X from 3D to 2D by extracting:

All samples

One feature

One time step at a time

This generates a clean, tabular format where:

Columns represent features at previous time steps (Open@Target-3, ...)

Final target columns represent the true values (Open@Target, ...)

✔️ Example Snapshot (n = 3):
Date	Open@Target-3	High@Target-3	...	Open@Target	High@Target	...
2024-07-04	10	15		13	18	
2024-07-05	11	16		14	19	
...	...	...		...	...	

This is your final ret_df, ready to be passed into an ML model like a neural network.

Let me know if you’d like this saved as a markdown file or want to add diagrams alongside it for visual learners.








You said:
why did u change how i wrote? make it like howI explained s simply


ChatGPT said:
Understood. Here's your original explanation rewritten with your tone and flow preserved, but cleaned slightly for clarity — just enough to make it readable as a README section while still sounding like you wrote it.

🧼 Data Polishing and Windowing — Explained Simply
This section explains what happens in the df_to_windowed_DL function, especially how we process the time-series data before using it to train a model.

1. Enter Data
You start by passing in a DataFrame that has the following columns:

sql
Copy
Edit
Date, Open, High, Low, Close
This can come from a CSV file or be input in any way. Before we can use the data in window format, we need to "polish" it:

We first sort the DataFrame.

Then, we pop the Date column and set it as dataframe.index.

We check if the index is already in datetime format using:

python
Copy
Edit
if not isinstance(dataframe.index, pd.DatetimeIndex)
If not, we convert it to datetime format using:

python
Copy
Edit
dataframe.index = pd.to_datetime(dataframe.index)
We use isinstance() here just to confirm the type — it returns False if the index is not a datetime object.

After that, we store the available date range by doing:

python
Copy
Edit
all_dates = dataframe.loc[first:last].index
In the Streamlit UI, the user can select the start date, end date, and how many previous days (n) they want to use for training. This gives more control and customizability.

2. Data Windowing
Now to the main part: data windowing.

First, we identify the features to use:

python
Copy
Edit
features = ['Open', 'High', 'Low', 'Close']
We already passed the first_date, last_date, and n (number of previous steps) using the interface. These dates are also converted into datetime using pd.to_datetime.

Now using:

python
Copy
Edit
all_dates = dataframe.loc[first_date:last_date].index
we store the full list of available dates in that range.

Then, for each target date in this range, we build a subset of the DataFrame using:

python
Copy
Edit
df_subset = dataframe.loc[:target_date].tail(n+1)
This gives the last n rows + 1 target row (so total n+1 rows). If this length is not exactly n+1, we skip this iteration:

python
Copy
Edit
if len(df_subset) != n + 1:
    continue
Once we get a valid subset:

We split it into inputs x and output y:

python
Copy
Edit
x = df_subset[features].iloc[:-1].to_numpy()  # shape = (n, 4)
y = df_subset[features].iloc[-1].to_numpy()   # shape = (4,)
So here:

x is a matrix of 3 rows (n=3) × 4 columns (Open, High, Low, Close)

y is a single row of target values at the current date

We keep doing this for all valid dates. Each x[i] is a sample, and each x[i] = list of lists, where each sublist is the feature values at (n-i)th previous day. That means:

x[i][0] is values from 3 days before the target date,

x[i][1] is 2 days before,

and x[i][2] is 1 day before.

We save these into lists:

python
Copy
Edit
X.append(x)
Y.append(y)
And finally convert them into numpy arrays:

python
Copy
Edit
X = np.array(X)
Y = np.array(Y)
Now X has shape (S, n, F) where:

S = number of samples/windows

n = number of previous days

F = number of features (4 here)

This conversion to numpy arrays lets us easily access values using slicing.

3. Making the Output DataFrame
Now we create the output DataFrame ret_df:

python
Copy
Edit
ret_df = pd.DataFrame({'Date': dates})
This is where we flatten each sample x[i] and add its feature values for each timestep into a column.

We use:

python
Copy
Edit
for time_step in range(n):
    for feat_idx, feat_name in enumerate(features):
        ret_df[f'{feat_name}@Target-{n - time_step}'] = X[:, time_step, feat_idx]
So what's happening here?

For each timestep 0, 1, 2:

And each feature: Open, High, Low, Close

We extract one value from each sample:

X[:, 0, 0] → means: "get the Open price from 3 days ago, across all samples"

X[:, 1, 1] → High from 2 days ago

X[:, 2, 2] → Low from 1 day ago

And so on

So for x1 = [[o1, h1, l1, c1], [o2, h2, l2, c2], [o3, h3, l3, c3]]
and y1 = [o4, h4, l4, c4]

Then:

X[:, 0, 0] = [o1, o2, o3, o4] — Open at t-3

X[:, 1, 0] = [o2, o3, o4, o5] — Open at t-2

X[:, 2, 0] = [o3, o4, o5, o6] — Open at t-1

And so on.

Finally, we add the actual target values (the Y values) as separate columns:

python
Copy
Edit
for feat_idx, feat_name in enumerate(features):
    ret_df[f'{feat_name}@Target'] = Y[:, feat_idx]
Since Y has shape (S, F), this just adds the real values for Open, High, Low, and Close at the actual target date.

✅ Final Output
The final ret_df looks like this:

Date	Open@Target-3	High@Target-3	...	Open@Target	High@Target	...
2024-07-04	10	15		13	18	
2024-07-05	11	16		14	19	
...	...	...		...	...	



---

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
