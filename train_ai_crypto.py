# train_ai_crypto.py
# RL agent for Sharia-compliant Crypto trading (same logic as train_ai.py, but for crypto assets only)

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import deque
import random
import requests
import time
import os

# Load datasets
with open('asset_screening_dataset.json') as f:
    asset_screening = json.load(f)
with open('trading_strategies_dataset.json') as f:
    trading_strategies = json.load(f)
with open('market_patterns_dataset.json') as f:
    market_patterns = json.load(f)

# Filter for Sharia-compliant crypto assets (case-insensitive)
crypto_assets = [a for a in asset_screening if a.get('asset_class', '').lower() == 'crypto' and a.get('halal_status')]

# RL agent parameters
STATE_SIZE = 16  # Adjust as needed for your state vector
ACTION_SIZE = len(crypto_assets) * len(trading_strategies)
MEMORY_SIZE = 10000
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
WAIT_PERIOD = 5
MAX_TRADES = 3

# Experience replay buffer
memory = deque(maxlen=MEMORY_SIZE)

# Build Q-network
model = keras.Sequential([
    keras.layers.Dense(64, input_dim=STATE_SIZE, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(ACTION_SIZE, activation='linear')
])
model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE))

# Helper: fetch historical chart data for a crypto symbol from LiveCoinWatch
def fetch_livecoinwatch_chart(symbol, start, end, api_key):
    """
    Fetch historical chart data for a crypto symbol from LiveCoinWatch.
    symbol: e.g. 'BTC', 'ETH'
    start, end: Unix timestamps
    api_key: your LiveCoinWatch API key
    Returns: JSON response with chart data
    """
    url = "https://api.livecoinwatch.com/coins/single/history"
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key
    }
    body = {
        "currency": "USD",
        "code": symbol,
        "start": start,
        "end": end,
        "meta": True
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"LiveCoinWatch API error: {response.status_code} {response.text}")
        return None

# Fetch historical chart data for a crypto symbol from LiveCoinWatch if chart_source is 'livecoinwatch'.
def fetch_crypto_chart_livecoinwatch(symbol, start, end, api_key):
    """
    Fetch historical chart data for a crypto symbol from LiveCoinWatch if chart_source is 'livecoinwatch'.
    symbol: e.g. 'BTC', 'ETH'
    start, end: Unix timestamps
    api_key: your LiveCoinWatch API key
    Returns: JSON response with chart data
    """
    url = "https://api.livecoinwatch.com/coins/single/history"
    headers = {
        "content-type": "application/json",
        "x-api-key": api_key
    }
    body = {
        "currency": "USD",
        "code": symbol,
        "start": start,
        "end": end,
        "meta": True
    }
    response = requests.post(url, headers=headers, json=body)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"LiveCoinWatch API error: {response.status_code} {response.text}")
        return None

# Example usage in your agent or chart logic:
# for asset in asset_screening:
#     if asset.get('asset_class', '').lower() == 'crypto' and asset.get('chart_source') == 'livecoinwatch':
#         chart = fetch_crypto_chart_livecoinwatch('BTC', start_timestamp, end_timestamp, YOUR_API_KEY)
#         # process chart data

# Helper: get GPT-like sentiment (stub, replace with your implementation)
def get_gpt_sentiment(state):
    # ...existing code for GPT sentiment...
    return 0.0, "Neutral"

# State vector builder (customize as needed)
def build_state(asset_idx, strategy_idx, chart_data, open_trades, profit_goal):
    # Extract price series from LiveCoinWatch chart data
    prices = []
    if chart_data and 'history' in chart_data and 'rate' in chart_data['history']:
        prices = chart_data['history']['rate']
    elif chart_data and 'rate' in chart_data:
        prices = chart_data['rate']
    else:
        prices = [0.0] * 30
    # Use last 30 prices (pad if needed)
    prices = prices[-30:] if len(prices) >= 30 else [0.0]*(30-len(prices)) + prices
    # Calculate returns and volatility
    returns = np.diff(prices).tolist() + [0.0]
    volatility = float(np.std(returns))
    avg_price = float(np.mean(prices))
    last_price = float(prices[-1])
    # Example state: [last_price, avg_price, volatility, profit_goal, len(open_trades)] + first 10 returns
    state = [last_price, avg_price, volatility, profit_goal, len(open_trades)] + returns[:10]
    # Pad to STATE_SIZE
    if len(state) < STATE_SIZE:
        state += [0.0] * (STATE_SIZE - len(state))
    return np.array(state[:STATE_SIZE])

# Crypto chart API key for chart access
CRYPTO_CHART_API_KEY = "4f2a44da-fe76-46d0-9608-e876d3b52659"

# Main training loop
def train_agent(episodes=100):
    global EPSILON
    profit_history = []
    for ep in range(episodes):
        total_profit = 0
        open_trades = []
        wait = 0
        if len(crypto_assets) == 0:
            print("No crypto assets available for training.")
            break
        asset_idx = random.randint(0, len(crypto_assets)-1)
        strategy_idx = random.randint(0, len(trading_strategies)-1)
        asset = crypto_assets[asset_idx]
        # Use LiveCoinWatch for chart data
        symbol = asset.get('crypto_id', 'BTC')
        # Example: last 7 days
        end = int(time.time())
        start = end - 7*24*60*60
        chart_data = fetch_crypto_chart_livecoinwatch(symbol, start, end, CRYPTO_CHART_API_KEY)
        if not chart_data:
            print(f"No chart data for {symbol}, skipping episode {ep+1}.")
            continue
        state = build_state(asset_idx, strategy_idx, chart_data, open_trades, profit_goal=0.05)
        done = False
        steps = 0
        while not done and steps < 100:
            if np.random.rand() < EPSILON:
                action = random.randrange(ACTION_SIZE)
            else:
                q_values = model.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(q_values[0])
            # ...existing code for action execution, trade management, reward calculation...
            # Store experience, train, update state, etc.
            done = True  # Replace with real episode termination logic
            steps += 1
        profit_history.append(total_profit)
        if EPSILON > EPSILON_MIN:
            EPSILON *= EPSILON_DECAY
        print(f"Episode {ep+1}: Profit {total_profit}")
    # Plot learning curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(profit_history, label='Profit per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.title('Learning Curve: Crypto RL Agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curve_crypto.png')
    plt.show()

if __name__ == '__main__':
    train_agent(episodes=100)
