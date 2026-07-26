"""
Food Delivery Time Prediction — Flask API (serving only)
Loads a pre-trained model (trained locally via train_local.py) from
model_cache/ and serves predictions + pre-generated graphs. No training
happens here, so this stays light enough for Render's free tier.
"""

import os
import pickle
import threading

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # quiet TensorFlow startup logs

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from tensorflow.keras.models import load_model

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
# Feature order — must match what train_local.py used while training.
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

STATE = {}
_LOAD_LOCK = threading.Lock()

# Paths for the artefacts trained locally and committed to the repo
CACHE_DIR = os.path.join(BASE_DIR, "model_cache")
MODEL_PATH = os.path.join(CACHE_DIR, "model.keras")
SCALER_PATH = os.path.join(CACHE_DIR, "scaler.pkl")
ENCODERS_PATH = os.path.join(CACHE_DIR, "encoders.pkl")
METADATA_PATH = os.path.join(CACHE_DIR, "metadata.pkl")


def _load_cache():
    """Load the pre-trained model artefacts (produced by train_local.py)."""
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, ENCODERS_PATH, METADATA_PATH]):
        print(
            "No model_cache/ found. Train the model locally first:\n"
            "  pip install -r requirements-train.txt\n"
            "  python train_local.py\n"
            "then commit the generated model_cache/ folder and redeploy."
        )
        return False
    try:
        print("Loading cached model from disk…")
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(ENCODERS_PATH, "rb") as f:
            encoders = pickle.load(f)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        model = load_model(MODEL_PATH)
        STATE["encoders"] = encoders
        STATE["scaler"] = scaler
        STATE["model"] = model
        STATE["metrics"] = metadata["metrics"]
        STATE["eda_graphs"] = metadata["eda_graphs"]
        STATE["model_graphs"] = metadata["model_graphs"]
        STATE["categorical_options"] = metadata["categorical_options"]
        print("Cache loaded successfully.", STATE["metrics"])
        return True
    except Exception as exc:
        print(f"Cache load failed ({exc}).")
        return False


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
        return jsonify({"error": "Model not loaded yet. Please retry in a few seconds."}), 503

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


def _startup():
    with _LOAD_LOCK:
        if STATE:
            return
        _load_cache()

_t = threading.Thread(target=_startup, daemon=True)
_t.start()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)