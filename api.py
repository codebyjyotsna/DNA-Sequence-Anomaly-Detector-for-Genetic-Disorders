from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from data_preprocessing import preprocess_sequences

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model("models/dna_anomaly_detector.h5")

@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict genetic disorder from DNA sequence.
    """
    data = request.json
    if "sequence" not in data:
        return jsonify({"error": "No DNA sequence provided"}), 400

    sequence = data["sequence"]
    preprocessed_sequence = preprocess_sequences([sequence])
    prediction = model.predict(preprocessed_sequence)
    result = {"sequence": sequence, "disorder_probability": float(prediction[0][0])}
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
