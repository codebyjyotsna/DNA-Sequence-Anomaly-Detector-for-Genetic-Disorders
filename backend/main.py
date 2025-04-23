from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()

# Load the pre-trained model
MODEL_PATH = "models/dna_anomaly_detector.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# DNA encoding map
DNA_ENCODING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# Define input schema
class DNASequence(BaseModel):
    sequence: str

def preprocess_sequence(sequence: str):
    """
    Preprocess DNA sequence into numerical encoding.
    """
    try:
        return np.array([[DNA_ENCODING[char] for char in sequence]])
    except KeyError:
        raise ValueError("Invalid character in DNA sequence. Allowed: A, C, G, T.")

@app.post("/predict")
def predict_dna_disorder(dna: DNASequence):
    """
    Predict the probability of a genetic disorder based on a DNA sequence.
    """
    try:
        processed_sequence = preprocess_sequence(dna.sequence)
        prediction = model.predict(processed_sequence)
        return {
            "sequence": dna.sequence,
            "disorder_probability": float(prediction[0][0])
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# Example root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the DNA Anomaly Detector API"}
