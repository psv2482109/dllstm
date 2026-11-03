# LSTM Time-Series Forecasting with Streamlit

A GitHub-ready educational project for training an LSTM on a time-series CSV, evaluating it on a chronological holdout period, and forecasting future values.

## Features
- Upload a CSV or use the included synthetic sample dataset.
- Select a numeric target and optional date/time column.
- Configure look-back window, forecast horizon, epochs, batch size, LSTM units, dropout, learning rate, and training percentage.
- Chronological train/test split; scaler is fitted on training data only.
- LSTM training with early stopping.
- Test metrics: RMSE, MAE, R².
- Actual-versus-predicted chart, recursive future forecast, and CSV download.

## Project structure
```text
LSTM_Streamlit_GitHub_Project/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .streamlit/config.toml
└── data/sample_timeseries.csv
```

## Run locally on Windows
Use Python 3.11 (recommended with the pinned TensorFlow package). Open a terminal in this folder:

```bash
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, usually http://localhost:8501.

## Push to GitHub
1. Extract the ZIP.
2. Create a GitHub repository, e.g. `lstm-streamlit-forecast`.
3. Upload the **contents** of this project folder to the repository root. `app.py` and `requirements.txt` must be visible at the root.
4. Or use Git:

```bash
git init
git add .
git commit -m "Add LSTM Streamlit forecasting app"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/lstm-streamlit-forecast.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## Deploy on Streamlit Community Cloud
1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Select **Create app**.
3. Choose your repository and the `main` branch.
4. Set the app file path to `app.py`.
5. Select **Deploy** and review app logs if startup fails.

Streamlit Community Cloud installs dependencies from `requirements.txt`. TensorFlow is a relatively large dependency, so initial installation may take longer and can be constrained by the deployment environment.

## CSV requirements
- One row per time step.
- At least 30 valid observations.
- One numeric target column, e.g. `sales`, `demand`, `price`, or `temperature`.
- Optional date/time column, e.g. `date`.
- Keep records chronological and avoid combining unrelated time series.

Example:
```csv
date,value
2024-01-01,120.5
2024-02-01,123.0
2024-03-01,121.8
```

## Modelling notes
- This is a learning/demo baseline, not a production forecasting system.
- The model uses a chronological holdout and fits the scaler on the training segment only.
- Multi-step forecasts are recursive; later forecasts depend on earlier predictions.
- Prediction intervals are not calculated.
- Compare against simple baselines and perform backtesting before using forecasts operationally.
- Do not use predictions for financial decisions without independent validation.

## Troubleshooting
- **TensorFlow installation fails:** use Python 3.11 locally and inspect Streamlit Cloud build logs. TensorFlow is large.
- **Not enough observations:** provide more rows or reduce the look-back window.
- **No numeric target:** convert the target column to numeric values before uploading.
- **App not updated:** verify repository, branch, and entrypoint (`app.py`).
