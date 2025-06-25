# train_ai_stocks.py
# RL agent for Sharia-compliant Stock trading (same logic as train_ai.py, but for stocks only)

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

# Finnhub API setup
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
FINNHUB_BASE = 'https://finnhub.io/api/v1'

# Filter for Sharia-compliant stock assets (case-insensitive)
stock_assets = [a for a in asset_screening if a.get('asset_class', '').lower() in ['stock', 'stocks'] and a.get('sharia_compliant')]

# Map to Finnhub stock symbols (e.g., 'AAPL', 'MSFT')
def get_finnhub_stock_symbol(asset):
    return asset.get('finnhub_symbol')

# RL agent parameters
STATE_SIZE = 16  # Adjust as needed for your state vector
ACTION_SIZE = len(stock_assets) * len(trading_strategies)
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

# Helper: fetch historical candles for a stock symbol
def fetch_stock_candles(symbol, resolution='D', count=200):
    url = f"{FINNHUB_BASE}/stock/candle?symbol={symbol}&resolution={resolution}&count={count}&token={FINNHUB_API_KEY}"
    r = requests.get(url)
    data = r.json()
    if data.get('s') != 'ok':
        return None
    return data

# Helper: get GPT-like sentiment (stub, replace with your implementation)
def get_gpt_sentiment(state):
    # ...existing code for GPT sentiment...
    return 0.0, "Neutral"

# State vector builder (customize as needed)
def build_state(asset_idx, strategy_idx, candles, open_trades, profit_goal):
    # ...existing code for state vector...
    return np.zeros(STATE_SIZE)

# Main training loop
def train_agent(episodes=100):
    global EPSILON
    profit_history = []
    for ep in range(episodes):
        total_profit = 0
        open_trades = []
        wait = 0
        if len(stock_assets) == 0:
            print("No stock assets available for training.")
            break
        asset_idx = random.randint(0, len(stock_assets)-1)
        strategy_idx = random.randint(0, len(trading_strategies)-1)
        asset = stock_assets[asset_idx]
        symbol = get_finnhub_stock_symbol(asset)
        candles = fetch_stock_candles(symbol)
        if not candles:
            print(f"No candles for {symbol}, skipping episode {ep+1}.")
            continue
        state = build_state(asset_idx, strategy_idx, candles, open_trades, profit_goal=0.05)
        done = False
        # Ensure at least one episode is run
        steps = 0
        while not done and steps < 100:
            if np.random.rand() < EPSILON:
                action = random.randrange(ACTION_SIZE)
            else:
                q_values = model.predict(state.reshape(1, -1), verbose=0)
                action = np.argmax(q_values[0])
            # ...existing code for action execution, trade management, reward calculation...
            # For brevity, see train_ai.py for full logic
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
    plt.title('Learning Curve: Stocks RL Agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curve_stocks.png')
    plt.show()

if __name__ == '__main__':
    train_agent(episodes=100)
