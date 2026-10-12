# SleepSense: Stress Level Predictor

Healthcare Data Analytics course project. Predicts stress level (3–8) from sleep, health and lifestyle data,
with a "Stress Coach" what-if simulator.

## Run locally
```
pip install -r requirements.txt
python train.py          # optional: retrains and rewrites models/ and data/
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a new GitHub repository under your account and push this folder's contents
   (`app.py`, `requirements.txt`, `data/`, `models/`, `.streamlit/`; `train.py` is optional).
2. Go to https://share.streamlit.io, sign in with GitHub, click **Create app**.
3. Pick the repository, branch `main`, main file `app.py`, then choose your own subdomain, e.g. `sleepsense-yourname`.
4. Deploy. The app is live at `https://<your-subdomain>.streamlit.app`.

Change the name shown in the app by editing `AUTHOR` at the top of `app.py`.
`train.py` reads `data/cleaned-dataset.csv` (the cleaned Kaggle dataset) and rewrites `models/` and `data/sleep_stress_data.csv`.
