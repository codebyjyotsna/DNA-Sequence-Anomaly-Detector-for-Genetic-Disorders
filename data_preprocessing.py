import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np

def load_data(file_path):
    """
    Load DNA sequence data from a file.
    """
    return pd.read_csv(file_path)

def preprocess_sequences(sequences):
    """
    Preprocess DNA sequences by encoding them.
    """
    # Example DNA encoding (A:0, C:1, G:2, T:3)
    encoding_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([[encoding_map[char] for char in seq] for seq in sequences])

def load_and_preprocess(file_path):
    """
    Load and preprocess DNA data.
    """
    data = load_data(file_path)
    sequences = preprocess_sequences(data['sequence'])
    labels = LabelEncoder().fit_transform(data['label'])
    return sequences, labels

if __name__ == "__main__":
    # Example usage
    sequences, labels = load_and_preprocess("data/dna_sequences.csv")
    print("Encoded sequences shape:", sequences.shape)
    print("Labels shape:", len(labels))
