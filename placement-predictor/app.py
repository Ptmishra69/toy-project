from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd

# -------------------- CONFIG --------------------
FEATURES = ["cgpa", "iq"]

# -------------------- APP INIT --------------------
app = Flask(__name__)
CORS(app)

# -------------------- LOAD MODEL --------------------
try:
    with open("model/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

except Exception as e:
    print("Error loading model or scaler:", e)
    model = None
    scaler = None

# -------------------- ROUTES --------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Placement Predictor API is running 🚀"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if not model or not scaler:
        return jsonify({"error": "Model not loaded properly"}), 500

    try:
        data = request.get_json()

        # Validate input
        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Extract features safely
        cgpa = float(data.get("cgpa", 0))
        iq   = float(data.get("iq", 0))

        # Create DataFrame (IMPORTANT: keeps feature names consistent)
        input_df = pd.DataFrame([[cgpa, iq]], columns=FEATURES)

        # Scale input
        scaled_input = scaler.transform(input_df)

        # Predict
        prediction  = model.predict(scaled_input)[0]
        probability = model.predict_proba(scaled_input)[0][1]

        return jsonify({
            "placement": int(prediction),
            "probability": round(float(probability) * 100, 2),
            "result": "Placed" if prediction == 1 else "Not Placed"
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# -------------------- MAIN --------------------
if __name__ == "__main__":
    app.run(debug=True, port=5000)