import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Masking
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# DNA encoding map
DNA_ENCODING = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def preprocess_sequences(sequences):
    """
    Preprocess DNA sequences by encoding them.
    """
    return np.array([[DNA_ENCODING[char] for char in seq] for seq in sequences])

# Load data
data = pd.read_csv("data/dna_sequences.csv")  # Replace with your dataset
sequences = preprocess_sequences(data['sequence'])
labels = LabelEncoder().fit_transform(data['label'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

# Define model
vocab_size = 4  # A, C, G, T
input_length = X_train.shape[1]

model = Sequential([
    Masking(mask_value=0, input_shape=(input_length,)),
    Embedding(input_dim=vocab_size, output_dim=128, input_length=input_length),
    LSTM(units=64, return_sequences=False),
    Dense(units=1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("models/dna_anomaly_detector.h5")
print("Model training complete and saved!")
