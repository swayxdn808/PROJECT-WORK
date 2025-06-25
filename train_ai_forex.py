import numpy as np
import tensorflow as tf
import json
import random
import requests
from collections import deque
import matplotlib.pyplot as plt

# Load datasets
with open('asset_screening_dataset.json') as f:
    sharia_assets = json.load(f)
ALLOWED_ASSETS = [a['asset_class'] for a in sharia_assets if a['asset_class'] == 'Forex']

with open('trading_strategies_dataset.json') as f:
    strategies = json.load(f)
STRATEGY_NAMES = [s['strategy'] for s in strategies]

with open('market_patterns_dataset.json') as f:
    market_patterns = json.load(f)

pattern_names = set()
for entry in market_patterns:
    pattern_names.update(entry.get('patterns', []))
pattern_names = sorted(list(pattern_names))
pattern_index = {name: i for i, name in enumerate(pattern_names)}

FINNHUB_API_KEY = "d1de2e9r01qn1ojn1cdgd1de2e9r01qn1ojn1ce0"
def fetch_finnhub_prices(symbol, resolution="1", count=100):
    import time
    from datetime import datetime
    end = int(time.mktime(datetime.now().timetuple()))
    start = end - count * 60
    url = f"https://finnhub.io/api/v1/forex/candle"
    params = {
        "symbol": symbol,
        "resolution": resolution,
        "from": start,
        "to": end,
        "token": FINNHUB_API_KEY
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        if data.get("s") == "ok":
            return data["c"]
    return [100.0] * count

# ...copy TradingEnv, QAgent, gpt_advisor, and training loop from train_ai.py, but set symbol_map = {"Forex": "OANDA:EUR_USD"} and only use Forex assets...

    plt.figure(figsize=(10,5))
    plt.plot(profit_history, label='Profit per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Profit')
    plt.title('Learning Curve: Forex RL Agent')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('learning_curve_forex.png')
    plt.show()
