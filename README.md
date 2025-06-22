# Oil AI – Streamlit Cloud

Lightweight demo of Oil AI (LSTM forecasting + risk alerts) using SQLite.
Deploy steps:

1. Fork this repo or upload to your GitHub.
2. Go to https://streamlit.io/cloud -> New app.
3. Select this repo, branch `main`, file `oil_ai_app.py`.
4. Add secrets:

```
OILAI_SECRET = "supersecret"
```

Upload a CSV/Excel with columns:

`Date, Oil_Production, Gas_Production, Water_Cut, Pressure, Temperature`
