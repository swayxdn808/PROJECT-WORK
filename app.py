from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load and preprocess the dataset (example: trading strategies)
def load_dataset():
    with open('trading_strategies_dataset.json') as f:
        data = json.load(f)
    texts = [item['description'] for item in data]
    labels = [item['strategy'] for item in data]
    return texts, labels

def prepare_model():
    texts, labels = load_dataset()
    tokenizer = Tokenizer(num_words=100)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=20)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=100, output_dim=16, input_length=20),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(len(set(y)), activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=100, verbose=0)
    return model, tokenizer, le

model, tokenizer, le = prepare_model()

HTML = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Trading Strategy Classifier</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        input, button { font-size: 1.1em; }
        .result { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h2>Trading Strategy Classifier</h2>
    <form id="form">
        <label for="desc">Enter a strategy description:</label><br>
        <input type="text" id="desc" name="desc" size="60" required>
        <button type="submit">Classify</button>
    </form>
    <div class="result" id="result"></div>
    <script>
        document.getElementById('form').onsubmit = async function(e) {
            e.preventDefault();
            const desc = document.getElementById('desc').value;
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ desc })
            });
            const data = await res.json();
            document.getElementById('result').innerText = 'Predicted strategy: ' + data.strategy;
        };
    </script>
</body>
</html>
'''

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    desc = request.json.get('desc', '')
    seq = tokenizer.texts_to_sequences([desc])
    X = pad_sequences(seq, maxlen=20)
    pred = model.predict(X)
    label = le.inverse_transform([np.argmax(pred)])[0]
    return jsonify({"strategy": label})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
