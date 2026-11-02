# Graduate Admission Prediction — Logistic Regression + Streamlit

A Streamlit app that estimates the probability of admission to a graduate programme from **GRE score, university
rating and CGPA**, using a statsmodels logistic regression whose features were chosen with recursive feature
elimination (RFE). Decision threshold: 0.6.

## Model

- `logreg_rfe_model.pkl` — statsmodels `Logit` results object, fitted on 320 rows (pseudo-R² 0.633).
- Coefficients (standardised inputs): const −0.42, GRE 0.85, University Rating 0.88, CGPA 2.86 — CGPA is the strongest
  predictor.

## ⚠️ Known issue — input scaling

The model was fitted on **standardised** features, but the original app passed raw values (e.g. GRE = 320), which
made every prediction "Admit" with probability 1.0. The app now standardises inputs with the training means and
standard deviations from `scaler.json`, and shows an error instead of a prediction when that file is missing.

**To finish the fix:** from the training notebook, export the mean and standard deviation of `GRE Score`,
`University Rating` and `CGPA` on the *training* split into `scaler.json` (format: `scaler.example.json`) and commit it.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or open the repository in a GitHub Codespace — the dev container installs the requirements and starts the app on
port 8501.

## Files

```
app.py                 Streamlit app
logreg_rfe_model.pkl   fitted statsmodels model
scaler.example.json    format of the required scaler.json
requirements.txt
.devcontainer/         Codespaces / VS Code dev container
```
