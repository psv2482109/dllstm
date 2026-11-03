import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM

st.set_page_config(page_title="LSTM Time-Series Forecast", page_icon="📈", layout="wide")
st.title("📈 LSTM Time-Series Forecasting")
st.write("Upload a time-series CSV, train an LSTM model, evaluate it on a chronological holdout set, and forecast future values.")
st.info("Educational demo only. Forecasts are not financial advice. Validate data quality, leakage controls, and model performance before operational use.")


def make_sequences(values, window, start_target, end_target):
    xs, ys = [], []
    for target in range(start_target, end_target):
        xs.append(values[target-window:target])
        ys.append(values[target])
    if not xs:
        return np.empty((0, window, 1), dtype=np.float32), np.empty((0, 1), dtype=np.float32)
    return np.asarray(xs, dtype=np.float32).reshape(-1, window, 1), np.asarray(ys, dtype=np.float32).reshape(-1, 1)


def build_model(window, units, dropout, learning_rate):
    model = Sequential([
        Input(shape=(window, 1)),
        LSTM(units),
        Dropout(dropout),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model


def future_dates(date_values, horizon):
    parsed = pd.to_datetime(date_values, errors="coerce").dropna().sort_values()
    if len(parsed) < 2:
        return pd.date_range(pd.Timestamp.today().normalize() + pd.Timedelta(days=1), periods=horizon, freq="D")
    freq = pd.infer_freq(parsed)
    if freq:
        return pd.date_range(parsed.iloc[-1], periods=horizon + 1, freq=freq)[1:]
    step = parsed.diff().dropna().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(days=1)
    return pd.DatetimeIndex([parsed.iloc[-1] + step * i for i in range(1, horizon + 1)])


with st.sidebar:
    st.header("Data")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use included sample data", value=uploaded is None)
    st.divider()
    st.header("LSTM settings")
    window = st.slider("Look-back window", 5, 60, 12)
    horizon = st.slider("Forecast horizon", 1, 30, 7)
    epochs = st.slider("Maximum epochs", 5, 100, 25, step=5)
    batch_size = st.selectbox("Batch size", [8, 16, 32, 64], index=1)
    units = st.selectbox("LSTM units", [16, 32, 64, 128], index=1)
    dropout = st.slider("Dropout", 0.0, 0.5, 0.1, 0.05)
    learning_rate = st.select_slider("Learning rate", options=[0.0001, 0.0005, 0.001, 0.005], value=0.001)
    train_pct = st.slider("Training data (%)", 60, 90, 80, 5)

try:
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
        source = uploaded.name
    elif use_sample:
        raw = pd.read_csv("data/sample_timeseries.csv")
        source = "data/sample_timeseries.csv"
    else:
        st.warning("Upload a CSV or select the sample dataset.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read CSV: {exc}")
    st.stop()

if raw.empty:
    st.error("The CSV contains no data rows.")
    st.stop()

st.subheader("Dataset preview")
st.caption(f"Source: {source} | Rows: {len(raw):,} | Columns: {len(raw.columns)}")
st.dataframe(raw.head(10), use_container_width=True)
columns = list(raw.columns)
numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(raw[c])]
if not numeric_cols:
    st.error("No numeric columns found. Convert the target column to numbers and upload again.")
    st.stop()
target_default = numeric_cols[0]
date_default = next((c for c in columns if "date" in c.lower() or "time" in c.lower()), "(No date column)")
target = st.selectbox("Numeric target column", numeric_cols, index=numeric_cols.index(target_default))
date_options = ["(No date column)"] + columns
date_index = date_options.index(date_default) if date_default in date_options else 0
date_col = st.selectbox("Date/time column (optional)", date_options, index=date_index)

work = raw.copy()
if date_col != "(No date column)":
    work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
    work = work.dropna(subset=[date_col]).sort_values(date_col)
    time_values = work[date_col].reset_index(drop=True)
else:
    time_values = pd.Series(range(len(work)), name="Time step")
series = pd.to_numeric(work[target], errors="coerce")
mask = series.notna()
series = series[mask].astype(float).reset_index(drop=True)
time_values = time_values[mask.reset_index(drop=True)].reset_index(drop=True)
if len(series) < max(30, window + 10):
    st.error(f"Only {len(series)} valid observations. Provide at least {max(30, window + 10)} rows or reduce the look-back window.")
    st.stop()
if not np.isfinite(series.to_numpy()).all():
    st.error("Target column contains infinite values. Clean the data and upload again.")
    st.stop()

st.metric("Valid observations", f"{len(series):,}")
history = pd.DataFrame({"Time": time_values, target: series})
st.subheader("Historical series")
st.line_chart(history.set_index("Time"), use_container_width=True)

if st.button("🚀 Train LSTM and Forecast", type="primary", use_container_width=True):
    tf.keras.utils.set_random_seed(42)
    array = series.to_numpy(dtype=np.float32).reshape(-1, 1)
    split = int(len(array) * train_pct / 100)
    if split <= window or len(array) - split < 2:
        st.error("Not enough training or test observations. Change the training percentage or look-back window.")
        st.stop()

    # Fit scaling parameters on training data only to reduce test-data leakage.
    scaler = MinMaxScaler()
    scaler.fit(array[:split])
    scaled = scaler.transform(array).astype(np.float32)
    x_train, y_train = make_sequences(scaled, window, window, split)
    x_test, y_test = make_sequences(scaled, window, split, len(scaled))
    if len(x_train) < 5 or len(x_test) < 1:
        st.error("Not enough sequences. Reduce look-back window or adjust training percentage.")
        st.stop()

    model = build_model(window, units, dropout, learning_rate)
    early = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    bar = st.progress(0, text="Training LSTM...")
    try:
        training = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                             validation_split=0.15, shuffle=False, callbacks=[early], verbose=0)
        bar.progress(70, text="Evaluating test data...")
        pred_scaled = model.predict(x_test, verbose=0)
        actual = scaler.inverse_transform(y_test).ravel()
        predicted = scaler.inverse_transform(pred_scaled).ravel()
        rmse = float(np.sqrt(mean_squared_error(actual, predicted)))
        mae = float(mean_absolute_error(actual, predicted))
        r2 = float(r2_score(actual, predicted)) if len(actual) > 1 else float("nan")

        current = scaled[-window:].reshape(1, window, 1).copy()
        future_scaled = []
        for _ in range(horizon):
            next_value = float(model.predict(current, verbose=0)[0, 0])
            future_scaled.append(next_value)
            current = np.concatenate([current[:, 1:, :], np.array(next_value, dtype=np.float32).reshape(1, 1, 1)], axis=1)
        future_values = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).ravel()
        bar.progress(100, text="Completed")
        st.success("LSTM training and forecasting completed.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Test RMSE", f"{rmse:,.4f}")
        c2.metric("Test MAE", f"{mae:,.4f}")
        c3.metric("Test R²", f"{r2:,.4f}" if np.isfinite(r2) else "N/A")

        test_dates = time_values.iloc[split:].reset_index(drop=True)
        test_df = pd.DataFrame({"Time": test_dates, "Actual": actual, "Predicted": predicted})
        st.subheader("Test period: actual vs predicted")
        st.line_chart(test_df.set_index("Time"), use_container_width=True)

        if date_col != "(No date column)":
            future_time = future_dates(time_values, horizon)
        else:
            future_time = np.arange(len(series), len(series) + horizon)
        forecast = pd.DataFrame({"Time": future_time, "Forecast": future_values})
        st.subheader("Future forecast")
        st.dataframe(forecast, use_container_width=True)
        st.line_chart(forecast.set_index("Time"), use_container_width=True)

        export = pd.concat([
            pd.DataFrame({"Record_Type": "Test Prediction", "Time": test_df["Time"].astype(str),
                          "Actual": test_df["Actual"], "Predicted_or_Forecast": test_df["Predicted"]}),
            pd.DataFrame({"Record_Type": "Future Forecast", "Time": forecast["Time"].astype(str),
                          "Actual": np.nan, "Predicted_or_Forecast": forecast["Forecast"]}),
        ], ignore_index=True)
        st.download_button("Download predictions and forecast CSV", export.to_csv(index=False).encode("utf-8"),
                           "lstm_predictions_and_forecast.csv", "text/csv")
        with st.expander("Training history"):
            hist_df = pd.DataFrame(training.history)
            st.line_chart(hist_df[["loss", "val_loss"]])
            st.dataframe(hist_df.tail(), use_container_width=True)
        st.caption("Forecast is recursive: each prediction is used as input to the next step. Prediction intervals are not calculated.")
    except Exception as exc:
        bar.empty()
        st.exception(exc)
else:
    st.markdown("""
    **How to use**
    1. Upload a CSV with one numeric time-series target.
    2. Optionally choose a date/time column.
    3. Set look-back window, forecast horizon, and training parameters in the sidebar.
    4. Click **Train LSTM and Forecast**.
    5. Review test metrics, actual-versus-predicted results, and future forecasts.
    """)

with st.expander("CSV format example"):
    st.code("date,value\\n2024-01-01,120.5\\n2024-02-01,123.0\\n2024-03-01,121.8", language="csv")
