import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Masking
from sklearn.model_selection import train_test_split

def build_model(input_length, vocab_size, embedding_dim=128, lstm_units=64):
    """
    Build the RNN/LSTM model for DNA sequence analysis.
    """
    model = Sequential([
        Masking(mask_value=0, input_shape=(input_length,)),
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length),
        LSTM(units=lstm_units, return_sequences=False),
        Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    # Example usage
    from data_preprocessing import load_and_preprocess

    sequences, labels = load_and_preprocess("data/dna_sequences.csv")
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=0.2, random_state=42)

    vocab_size = 4  # A, C, G, T
    input_length = X_train.shape[1]

    model = build_model(input_length=input_length, vocab_size=vocab_size)
    model.summary()

    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
