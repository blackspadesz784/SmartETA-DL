# Food Delivery Time Predictor — Deep Learning (Keras ANN) Web App

This turns your `DL_Food_Delivery_Time_Prediction_Project.ipynb` notebook into
a working web app: a Flask backend that runs the same data-cleaning →
feature-engineering pipeline and trains the same Keras neural network
(Dense 128 → 64 → 32 → 1), plus a browser dashboard showing **every graph**
from the notebook and a live prediction form.

```
food_delivery_dl_app/
├── backend/
│   ├── app.py           ← Flask API + data pipeline + Keras ANN training
│   ├── requirements.txt
│   └── train.csv         ← your dataset (already copied in)
└── frontend/
    └── index.html         ← the dashboard (served by Flask, no build step)
```

## 1. Requirements

- Python 3.9–3.12
- `pip`
- ~2 GB free RAM (TensorFlow)

## 2. Install dependencies

```bash
cd food_delivery_dl_app/backend
pip install -r requirements.txt
```

TensorFlow is a large package — this install can take a few minutes the
first time.

## 3. Run the app

```bash
python app.py
```

You'll see:

```
Training the neural network on startup, this can take 1-3 minutes...
Training ANN for up to 120 epochs (early stopping enabled)...
Training complete. {'mae': ..., 'r2': ...}
 * Running on http://127.0.0.1:5001
```

Training happens once on startup (not per request). On a normal laptop CPU
this takes **~1.5–3 minutes** — early stopping halts training automatically
once validation loss stops improving, so it usually finishes well before the
full epoch budget.

## 4. Open the dashboard

Just **double-click `frontend/index.html`** — it opens directly in your
browser and talks to the backend at `http://localhost:5001` automatically
(no need to open it through Flask). Keep the `app.py` terminal running in
the background while you use it.

Three tabs:

- **Data Overview** — correlation heatmap, feature distributions, boxplots,
  age-vs-time scatter, orders-by-city countplot (same EDA as the notebook).
- **Model & Training** — the network architecture, R² / MAE / RMSE, the
  train-vs-validation loss curve, the actual-vs-predicted scatter plot, and
  the actual-vs-predicted bar chart for the first 20 test samples.
- **Try a Prediction** — fill in order details and get an instant delivery
  time estimate from the trained network.

## 5. Using a different dataset later

Replace `backend/train.csv` with a new file (same column names) and restart
`python app.py` — it re-trains automatically on startup.

## Notes on differences from the raw notebook

- **Epochs**: the notebook trains for a fixed 200 epochs. This app trains
  for up to 120 epochs but with **early stopping** (stops once validation
  loss stops improving for 15 epochs, keeping the best weights) — this keeps
  startup time reasonable without hurting accuracy; feel free to raise
  `EPOCHS` in `app.py` if you want to match the notebook exactly.
- Same data-cleaning fixes as the ML version were needed here too: trailing
  spaces / `"NaN "` strings in category columns, the `"conditions "` prefix
  in `Weatherconditions`, and the target column `Time_taken(min)` stored as
  text like `"(min) 24"` — all now parsed correctly before training.
- Everything else (feature list and order, train/test split, scaling,
  network architecture, loss function) mirrors your notebook exactly.

## Troubleshooting

- **Port 5001 already in use** → edit the last line of `app.py`, change
  `port=5001` to something else, **and** update the `API` constant near the
  top of the `<script>` section in `frontend/index.html` to match (e.g.
  `const API = 'http://localhost:5050';`) so the two stay in sync.
- **TensorFlow install fails / very slow** → make sure you're on Python
  3.9–3.12 (TensorFlow doesn't yet support every Python version); a fresh
  virtual environment usually helps: `python -m venv venv && source
  venv/bin/activate` (Windows: `venv\Scripts\activate`) before installing.
- **Training feels stuck** → it isn't — Keras trains silently in this app
  (no per-epoch printout) to keep the terminal clean; give it the full
  1–3 minutes shown in the startup message.
- **Graphs don't load / "waiting for backend"** → confirm `app.py` is still
  running and has printed "Training complete".
