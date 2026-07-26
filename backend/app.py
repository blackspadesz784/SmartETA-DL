"""
Food Delivery Time Prediction — Deep Learning Backend
Replicates the preprocessing + Keras ANN training pipeline from the
DL_Food_Delivery_Time_Prediction_Project notebook and exposes it as a
Flask API so a web frontend can show every graph and get live predictions.
"""

import os
import io
import base64
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # quiet TensorFlow startup logs

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=os.path.abspath(os.path.join(BASE_DIR, "..")), static_url_path="")
CORS(app, resources={r"/*": {"origins": "*"}})

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, PUT, DELETE"
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify({"error": str(e)}), 500


# ----------------------------------------------------------------------
# Feature order — same pipeline/order as the notebook produces.
# ----------------------------------------------------------------------
FEATURE_ORDER = [
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Weatherconditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
    "Day",
    "Month",
    "Order_Time",
    "Pickup_Time",
]

CATEGORICAL_COLS = [
    "Weatherconditions",
    "Road_traffic_density",
    "Type_of_order",
    "Type_of_vehicle",
    "Festival",
    "City",
]

EPOCHS = 120        # notebook used 200; capped + EarlyStopping for a
BATCH_SIZE = 32      # reasonable startup time in a demo web app
STATE = {}


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning as the notebook, made robust to the raw file's quirks
    (trailing spaces, 'NaN ' strings, '(min) 24' target format)."""
    df = df.copy()
    df = df.drop_duplicates()

    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in text_cols:
        df[col] = df[col].astype(str).str.strip()

    # "conditions Sunny" -> "Sunny" (before NaN normalisation — the raw file
    # has a literal "conditions NaN" value that must be caught as missing)
    df["Weatherconditions"] = df["Weatherconditions"].str.replace("conditions ", "", regex=False)

    for col in text_cols:
        df[col] = df[col].replace({"NaN": np.nan, "nan": np.nan})

    for col in ["Delivery_person_Age", "Delivery_person_Ratings", "multiple_deliveries"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    df = df.drop(["ID", "Delivery_person_ID"], axis=1)

    df["Order_Date"] = pd.to_datetime(df["Order_Date"], dayfirst=True)
    df["Day"] = df["Order_Date"].dt.day
    df["Month"] = df["Order_Date"].dt.month
    df = df.drop("Order_Date", axis=1)

    df["Time_Orderd"] = pd.to_datetime(df["Time_Orderd"], format="%H:%M:%S", errors="coerce")
    df["Time_Order_picked"] = pd.to_datetime(df["Time_Order_picked"], format="%H:%M:%S", errors="coerce")
    df["Time_Orderd"] = df["Time_Orderd"].fillna(df["Time_Orderd"].mode()[0])
    df["Time_Order_picked"] = df["Time_Order_picked"].fillna(df["Time_Order_picked"].mode()[0])
    df["Order_Time"] = df["Time_Orderd"].dt.hour * 60 + df["Time_Orderd"].dt.minute
    df["Pickup_Time"] = df["Time_Order_picked"].dt.hour * 60 + df["Time_Order_picked"].dt.minute
    df = df.drop(["Time_Orderd", "Time_Order_picked"], axis=1)

    # Target arrives as "(min) 24" -> 24
    df["Time_taken(min)"] = df["Time_taken(min)"].astype(str).str.extract(r"(\d+)").astype(float)

    return df


def make_eda_graphs(df_numeric_view, city_series, age_series, time_series):
    graphs = {}

    fig = plt.figure(figsize=(15, 10))
    sns.heatmap(df_numeric_view.corr(), annot=True, cmap="coolwarm", annot_kws={"size": 7})
    plt.title("Correlation Heatmap")
    graphs["heatmap"] = fig_to_base64(fig)

    fig = df_numeric_view.hist(figsize=(18, 15))
    plt.suptitle("Feature Distributions")
    graphs["histograms"] = fig_to_base64(plt.gcf())

    n_cols = df_numeric_view.shape[1]
    ncols = 4
    nrows = int(np.ceil(n_cols / ncols))
    df_numeric_view.plot(kind="box", subplots=True, layout=(nrows, ncols), figsize=(18, nrows * 3.2))
    plt.suptitle("Boxplots (outlier check)")
    graphs["boxplots"] = fig_to_base64(plt.gcf())

    fig = plt.figure(figsize=(8, 5))
    sns.scatterplot(x=age_series, y=time_series)
    plt.xlabel("Delivery_person_Age")
    plt.ylabel("Time_taken(min)")
    graphs["scatter_age_time"] = fig_to_base64(fig)

    fig = plt.figure(figsize=(8, 5))
    sns.countplot(x=city_series)
    plt.xlabel("City")
    graphs["countplot_city"] = fig_to_base64(fig)

    return graphs


def train_pipeline():
    csv_path = os.path.join(BASE_DIR, "train.csv")
    if not os.path.exists(csv_path):
        csv_path = "train.csv"
    raw = pd.read_csv(csv_path)
    df = clean_dataframe(raw)

    eda_numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    df_numeric_for_eda = df[eda_numeric_cols]
    city_for_eda = df["City"]
    age_for_eda = df["Delivery_person_Age"]
    time_for_eda = df["Time_taken(min)"]

    eda_graphs = make_eda_graphs(df_numeric_for_eda, city_for_eda, age_for_eda, time_for_eda)

    encoders = {}
    for col in CATEGORICAL_COLS:
        enc = LabelEncoder()
        df[col] = enc.fit_transform(df[col].astype(str))
        encoders[col] = enc

    X = df[FEATURE_ORDER]
    y = df["Time_taken(min)"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = Sequential([
        Dense(128, activation="relu", input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation="relu"),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

    print(f"Training ANN for up to {EPOCHS} epochs (early stopping enabled)...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=0,
    )

    prediction = model.predict(X_test_scaled, verbose=0).flatten()
    y_test_arr = np.array(y_test).flatten()

    mae = float(mean_absolute_error(y_test_arr, prediction))
    mse = float(mean_squared_error(y_test_arr, prediction))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test_arr, prediction))

    model_graphs = {}

    # Training / validation loss curve
    fig = plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Training History")
    plt.legend()
    model_graphs["loss_curve"] = fig_to_base64(fig)

    # Actual vs Predicted scatter
    fig = plt.figure(figsize=(8, 5))
    plt.scatter(y_test_arr, prediction, alpha=0.4, color="royalblue")
    lims = [min(y_test_arr.min(), prediction.min()), max(y_test_arr.max(), prediction.max())]
    plt.plot(lims, lims, "--", color="gray")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    model_graphs["scatter_actual_vs_predicted"] = fig_to_base64(fig)

    # Actual vs Predicted bar chart (first 20 samples)
    n = 20
    x = np.arange(n)
    width = 0.35
    fig = plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, y_test_arr[:n], width, label="Actual")
    plt.bar(x + width / 2, prediction[:n], width, label="Predicted")
    plt.xlabel("Test sample #")
    plt.ylabel("Delivery Time (min)")
    plt.title("Actual vs Predicted (first 20 test samples)")
    plt.xticks(x)
    plt.legend()
    model_graphs["bar_actual_vs_predicted"] = fig_to_base64(fig)

    STATE["encoders"] = encoders
    STATE["scaler"] = scaler
    STATE["model"] = model
    STATE["metrics"] = {
        "mae": round(mae, 3), "mse": round(mse, 3),
        "rmse": round(rmse, 3), "r2": round(r2, 4),
        "epochs_run": len(history.history["loss"]),
    }
    STATE["eda_graphs"] = eda_graphs
    STATE["model_graphs"] = model_graphs
    STATE["categorical_options"] = {
        col: sorted(encoders[col].classes_.tolist()) for col in CATEGORICAL_COLS
    }
    print("Training complete.", STATE["metrics"])


# ----------------------------------------------------------------------
# API routes
# ----------------------------------------------------------------------

@app.route("/api/status")
def status():
    return jsonify({"ready": bool(STATE), "metrics": STATE.get("metrics")})


@app.route("/api/eda-graphs")
def eda_graphs():
    return jsonify(STATE.get("eda_graphs", {}))


@app.route("/api/model-graphs")
def model_graphs():
    return jsonify(STATE.get("model_graphs", {}))


@app.route("/api/metrics")
def metrics():
    return jsonify(STATE.get("metrics", {}))


@app.route("/api/options")
def options():
    return jsonify({
        "categorical": STATE.get("categorical_options", {}),
        "vehicle_condition_range": [0, 1, 2, 3],
        "multiple_deliveries_range": [0, 1, 2, 3],
    })


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return "", 200

    if not STATE or "encoders" not in STATE or "model" not in STATE:
        return jsonify({"error": "Model training/initialization in progress. Please retry in a few seconds."}), 503

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No payload provided"}), 400

        row = {
            "Delivery_person_Age": float(data["Delivery_person_Age"]),
            "Delivery_person_Ratings": float(data["Delivery_person_Ratings"]),
            "Restaurant_latitude": float(data["Restaurant_latitude"]),
            "Restaurant_longitude": float(data["Restaurant_longitude"]),
            "Delivery_location_latitude": float(data["Delivery_location_latitude"]),
            "Delivery_location_longitude": float(data["Delivery_location_longitude"]),
            "Weatherconditions": STATE["encoders"]["Weatherconditions"].transform([data["Weatherconditions"]])[0],
            "Road_traffic_density": STATE["encoders"]["Road_traffic_density"].transform([data["Road_traffic_density"]])[0],
            "Vehicle_condition": int(data["Vehicle_condition"]),
            "Type_of_order": STATE["encoders"]["Type_of_order"].transform([data["Type_of_order"]])[0],
            "Type_of_vehicle": STATE["encoders"]["Type_of_vehicle"].transform([data["Type_of_vehicle"]])[0],
            "multiple_deliveries": int(data["multiple_deliveries"]),
            "Festival": STATE["encoders"]["Festival"].transform([data["Festival"]])[0],
            "City": STATE["encoders"]["City"].transform([data["City"]])[0],
            "Day": int(data["Day"]),
            "Month": int(data["Month"]),
            "Order_Time": int(data["Order_hour"]) * 60 + int(data["Order_minute"]),
            "Pickup_Time": int(data["Pickup_hour"]) * 60 + int(data["Pickup_minute"]),
        }
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error parsing input data: {str(e)}"}), 400

    try:
        X_input = pd.DataFrame([row])[FEATURE_ORDER]
        X_scaled = STATE["scaler"].transform(X_input)
        pred = float(STATE["model"].predict(X_scaled, verbose=0).flatten()[0])
        return jsonify({"prediction": round(pred, 1)})
    except Exception as e:
        return jsonify({"error": f"Prediction computation failed: {str(e)}"}), 500



@app.route("/health")
def health():
    return jsonify({"status": "ok", "ready": bool(STATE)})


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


if not STATE:
    print("Initializing neural network training pipeline on startup...")
    train_pipeline()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
