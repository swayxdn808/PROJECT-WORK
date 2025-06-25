from flask import Flask, request, jsonify, render_template_string, send_from_directory
import tensorflow as tf
import numpy as np
import json
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from train_ai import env, agent, INITIAL_BALANCE
import time
from datetime import datetime

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
    <title>AI Trading Profit Goal</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        input, button { font-size: 1.1em; }
        .result { margin-top: 20px; font-weight: bold; }
    </style>
</head>
<body>
    <h2>Set Profit Goal for AI Trading</h2>
    <form id="goalForm">
        <label for="profit_goal">Profit Goal ($):</label>
        <input type="number" id="profit_goal" name="profit_goal" required><br><br>
        <label for="period">Period (trading steps):</label>
        <input type="number" id="period" name="period" value="50" required><br><br>
        <button type="submit">Start Trading</button>
    </form>
    <div class="result" id="result"></div>
    <div id="gpt_summaries" style="margin-top:20px;"></div>
    <script>
        document.getElementById('goalForm').onsubmit = async function(e) {
            e.preventDefault();
            const profit_goal = document.getElementById('profit_goal').value;
            const period = document.getElementById('period').value;
            document.getElementById('result').innerText = 'Running...';
            document.getElementById('gpt_summaries').innerHTML = '';
            const res = await fetch('/trade', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ profit_goal, period })
            });
            const data = await res.json();
            document.getElementById('result').innerText = data.message;
            if (data.gpt_summaries) {
                document.getElementById('gpt_summaries').innerHTML = '<h3>GPT Advisor Summaries</h3><ul>' +
                    data.gpt_summaries.map(s => `<li>${s}</li>`).join('') + '</ul>';
            }
        };
    </script>
</body>
</html>
'''

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/trade", methods=["POST"])
def trade():
    req = request.get_json()
    profit_goal = float(req.get('profit_goal', 0))
    period = int(req.get('period', 50))
    state = env.reset()
    gpt_summaries = []
    for t in range(period):
        # Get GPT summary for this step
        _, gpt_summary = env.gpt_advisor(
            state, env.asset_idx, env.strategy_idx, env.pattern_entry
        ) if hasattr(env, 'gpt_advisor') else (0, "")
        gpt_summaries.append(f"Step {t+1}: {gpt_summary}")
        action, asset_idx, strategy_idx = agent.act(state)
        next_state, reward, done, _ = env.step(action, asset_idx, strategy_idx)
        agent.remember(state, action, asset_idx, strategy_idx, reward, next_state, done)
        state = next_state
        if done:
            break
        if env.balance - INITIAL_BALANCE >= profit_goal:
            break
    profit = env.balance - INITIAL_BALANCE
    message = f"Trading complete. Final balance: ${env.balance:.2f}. Profit: ${profit:.2f}. Goal: ${profit_goal:.2f}. {'Goal reached!' if profit >= profit_goal else 'Goal not reached.'}"
    # Add GPT summaries to the response
    return jsonify({"message": message, "gpt_summaries": gpt_summaries})

@app.route('/finnhub_chart')
def finnhub_chart():
    symbol = request.args.get('symbol', 'AAPL')
    FINNHUB_API_KEY = "d1de2e9r01qn1ojn1cdgd1de2e9r01qn1ojn1ce0"
    end = int(time.mktime(datetime.now().timetuple()))
    start = end - 60*60  # last 1 hour, 1-min bars
    url = f"https://finnhub.io/api/v1/stock/candle"
    params = {
        "symbol": symbol,
        "resolution": "1",
        "from": start,
        "to": end,
        "token": FINNHUB_API_KEY
    }
    resp = requests.get(url, params=params)
    closes = []
    times = []
    if resp.status_code == 200:
        data = resp.json()
        if data.get("s") == "ok":
            closes = data["c"]
            times = [datetime.fromtimestamp(ts).strftime('%H:%M') for ts in data["t"]]
    return jsonify({"closes": closes, "times": times})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
