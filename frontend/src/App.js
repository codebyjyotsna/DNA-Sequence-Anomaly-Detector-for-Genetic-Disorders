import React, { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [sequence, setSequence] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setResult(null);
    setError(null);

    if (sequence.trim() === "") {
      setError("DNA sequence cannot be empty!");
      return;
    }

    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", { sequence });
      setResult(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "An error occurred!");
    }
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>DNA Sequence Anomaly Detector</h1>
        <form onSubmit={handleSubmit}>
          <label>
            Enter DNA Sequence:
            <input
              type="text"
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder="e.g., ACGTACG..."
            />
          </label>
          <button type="submit">Predict</button>
        </form>
        {error && <div className="error">{error}</div>}
        {result && (
          <div className="result">
            <h2>Prediction Result</h2>
            <p><strong>Sequence:</strong> {result.sequence}</p>
            <p><strong>Disorder Probability:</strong> {result.disorder_probability.toFixed(4)}</p>
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
