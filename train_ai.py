import numpy as np
import tensorflow as tf
import json
import random
import requests
from collections import deque
import matplotlib.pyplot as plt

# Load Sharia-compliant asset classes
with open('asset_screening_dataset.json') as f:
    sharia_assets = json.load(f)
ALLOWED_ASSETS = [a['asset_class'] for a in sharia_assets]

# Load trading strategies
with open('trading_strategies_dataset.json') as f:
    strategies = json.load(f)
STRATEGY_NAMES = [s['strategy'] for s in strategies]

# Load market patterns dataset
with open('market_patterns_dataset.json') as f:
    market_patterns = json.load(f)

# Helper to extract features from market patterns
pattern_names = set()
for entry in market_patterns:
    pattern_names.update(entry.get('patterns', []))
pattern_names = sorted(list(pattern_names))
pattern_index = {name: i for i, name in enumerate(pattern_names)}

POLYGON_API_KEY = "RxEod5PIoWjRZS3jh1FoDVpiilFojDHk"

# Helper to fetch historical price data from Polygon.io
def fetch_polygon_prices(symbol, timespan="minute", limit=100):
    url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/{timespan}/2024-06-01/2024-06-30"
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": limit,
        "apiKey": POLYGON_API_KEY
    }
    resp = requests.get(url, params=params)
    if resp.status_code == 200:
        data = resp.json()
        if "results" in data:
            return [bar["c"] for bar in data["results"]]  # closing prices
    return [100.0] * limit  # fallback to flat prices if error

# Helper to fetch historical price data from Finnhub.io
FINNHUB_API_KEY = "d1de2e9r01qn1ojn1cdgd1de2e9r01qn1ojn1ce0"
def fetch_finnhub_prices(symbol, resolution="1", count=100):
    # symbol: e.g., 'AAPL', 'EURUSD', 'BINANCE:BTCUSDT'
    import time
    from datetime import datetime, timedelta
    end = int(time.mktime(datetime.now().timetuple()))
    start = end - count * 60  # 1-min bars, count minutes ago
    url = f"https://finnhub.io/api/v1/stock/candle"
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
            return data["c"]  # closing prices
    return [100.0] * count  # fallback to flat prices if error

LIVECOINWATCH_API_KEY = "4f2a44da-fe76-46d0-9608-e876d3b52659"

def fetch_livecoinwatch_prices(crypto_id="BTC", currency="USD", limit=100):
    url = "https://api.livecoinwatch.com/coins/single/history"
    headers = {
        'content-type': 'application/json',
        'x-api-key': LIVECOINWATCH_API_KEY
    }
    import time
    from datetime import datetime, timedelta
    now = datetime.utcnow()
    prices = []
    for i in range(limit):
        ts = int((now - timedelta(minutes=limit - i)).timestamp()) * 1000
        payload = {
            "currency": currency,
            "code": crypto_id,
            "start": ts,
            "end": ts + 60000,
            "meta": False
        }
        try:
            resp = requests.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                data = resp.json()
                price = data.get("rate", 100.0)
                prices.append(price)
            else:
                prices.append(100.0)
        except Exception:
            prices.append(100.0)
    return prices if prices else [100.0] * limit

# --- Simple custom GPT-like advisor using provided datasets ---
def gpt_advisor(state, asset_idx, strategy_idx, pattern_entry):
    """
    Simulates a GPT-like analysis using the datasets.
    Returns a sentiment score (-1: negative, 0: neutral, 1: positive) and a summary string.
    """
    asset = sharia_assets[asset_idx]
    strategy = strategies[strategy_idx]
    sector = pattern_entry.get('sector', '')
    tech = pattern_entry.get('technical_indicators', {})
    rsi = tech.get('rsi', 50)
    macd = tech.get('macd', 0)
    sma_50 = tech.get('sma_50', 0)
    sma_200 = tech.get('sma_200', 0)
    boll_upper = tech.get('bollinger_upper', 0)
    boll_lower = tech.get('bollinger_lower', 0)
    price = pattern_entry.get('price_data', {}).get('close', 0)
    patterns = pattern_entry.get('patterns', [])
    # Asset and strategy features
    n_criteria = len(asset.get('criteria', []))
    strat_desc = strategy.get('description', '')
    strat_notes = strategy.get('key_notes', '')
    # Expanded logic
    reasons = []
    sentiment = 0
    # Technicals
    if rsi > 70:
        sentiment -= 1
        reasons.append(f"RSI high ({rsi})")
    elif rsi < 30:
        sentiment += 1
        reasons.append(f"RSI low ({rsi})")
    if macd > 0:
        sentiment += 1
        reasons.append(f"MACD positive ({macd})")
    elif macd < 0:
        sentiment -= 1
        reasons.append(f"MACD negative ({macd})")
    if price > sma_50 > sma_200:
        sentiment += 1
        reasons.append(f"Price ({price}) > SMA50 ({sma_50}) > SMA200 ({sma_200})")
    elif price < sma_50 < sma_200:
        sentiment -= 1
        reasons.append(f"Price ({price}) < SMA50 ({sma_50}) < SMA200 ({sma_200})")
    if price > boll_upper:
        sentiment -= 1
        reasons.append(f"Price above Bollinger upper ({boll_upper})")
    elif price < boll_lower:
        sentiment += 1
        reasons.append(f"Price below Bollinger lower ({boll_lower})")
    # Pattern-based
    if 'bullish' in [p.lower() for p in patterns]:
        sentiment += 1
        reasons.append("Bullish pattern detected")
    if 'bearish' in [p.lower() for p in patterns]:
        sentiment -= 1
        reasons.append("Bearish pattern detected")
    # Asset/strategy-based
    if n_criteria > 3:
        sentiment += 1
        reasons.append(f"Asset has many Sharia criteria ({n_criteria})")
    if 'momentum' in strat_desc.lower() or 'momentum' in strat_notes.lower():
        if sentiment > 0:
            reasons.append("Momentum strategy aligns with bullish signals")
        elif sentiment < 0:
            reasons.append("Momentum strategy risky in bearish conditions")
    # Clamp sentiment
    sentiment = max(-1, min(1, sentiment))
    summary = f"{'Bullish' if sentiment==1 else 'Bearish' if sentiment==-1 else 'Neutral'}: {sector} sector. " + ", ".join(reasons)
    return sentiment, summary

# Simulated market environment with asset and strategy selection
class TradingEnv:
    def __init__(self, asset_classes, strategies, initial_balance=1000.0, max_positions=3, transaction_cost=2.0):
        self.asset_classes = asset_classes
        self.strategies = strategies
        # For LiveCoinWatch, use crypto_id from dataset or default to BTC
        self.crypto_id_map = {a['asset_class']: a.get('crypto_id', 'BTC') for a in sharia_assets}
        self.initial_balance = initial_balance
        self.max_positions = max_positions
        self.transaction_cost = transaction_cost
        self.reset()
    def reset(self):
        self.balance = self.initial_balance
        self.asset_idx = random.randrange(len(self.asset_classes))
        self.strategy_idx = random.randrange(len(self.strategies))
        self.asset = self.asset_classes[self.asset_idx]
        # Use LiveCoinWatch for all assets (or restrict to crypto if needed)
        crypto_id = self.crypto_id_map.get(self.asset, 'BTC')
        self.prices = fetch_livecoinwatch_prices(crypto_id=crypto_id)
        print("Fetched prices:", self.prices[:10])  # Debug: show first 10 prices
        self.price_ptr = 0
        self.price = self.prices[self.price_ptr]
        self.positions = []  # List of open positions: [{'entry': price, 'asset_idx': idx, 'strategy_idx': idx}]
        self.pattern_entry = random.choice(market_patterns)
        return self._get_state()
    def _get_state(self):
        gpt_sentiment, _ = gpt_advisor(
            None, self.asset_idx, self.strategy_idx, self.pattern_entry
        )
        # Add unrealized P&L and average entry price as features
        unrealized_pnl = sum(self.price - pos['entry'] for pos in self.positions) if self.positions else 0.0
        avg_entry = np.mean([pos['entry'] for pos in self.positions]) if self.positions else 0.0
        return np.concatenate((
            np.array([self.balance, self.price, len(self.positions), unrealized_pnl, avg_entry, self.asset_idx, self.strategy_idx]),
            self._get_asset_features(self.asset_idx),
            self._get_strategy_features(self.strategy_idx),
            self._get_pattern_features(self.pattern_entry),
            np.array([gpt_sentiment])
        ))
    def _get_asset_features(self, asset_idx):
        # Example: encode number of criteria and notes length
        asset = sharia_assets[asset_idx]
        num_criteria = len(asset.get('criteria', []))
        notes_length = len(asset.get('notes', ''))
        return np.array([num_criteria, notes_length])
    def _get_strategy_features(self, strategy_idx):
        # Example: encode description and key_notes length
        strat = strategies[strategy_idx]
        desc_length = len(strat.get('description', ''))
        notes_length = len(strat.get('key_notes', ''))
        return np.array([desc_length, notes_length])
    def _get_pattern_features(self, entry):
        # One-hot encode sector, timeframe, patterns, plus price/indicator features
        sector = entry.get('sector', '')
        sector_onehot = [1.0 if sector == s else 0.0 for s in ['finance', 'technology', 'energy']]
        timeframe = entry.get('timeframe', '')
        timeframe_onehot = [1.0 if timeframe == t else 0.0 for t in ['intraday', 'daily', 'weekly']]
        patterns = entry.get('patterns', [])
        patterns_onehot = [1.0 if name in patterns else 0.0 for name in pattern_names]
        price_data = entry.get('price_data', {})
        price_feats = [price_data.get(k, 0.0) for k in ['open', 'high', 'low', 'close', 'volume']]
        tech = entry.get('technical_indicators', {})
        tech_feats = [tech.get(k, 0.0) for k in ['sma_50', 'sma_200', 'rsi', 'macd', 'bollinger_upper', 'bollinger_lower']]
        return np.array(sector_onehot + timeframe_onehot + patterns_onehot + price_feats + tech_feats)
    def step(self, action, asset_idx, strategy_idx):
        self.asset_idx = asset_idx
        self.strategy_idx = strategy_idx
        self.asset = self.asset_classes[self.asset_idx]
        self.symbol = self.symbol_map.get(self.asset, "AAPL")
        reward = 0
        done = False
        # Only allow new trades if not at max positions and not in a waiting period
        if not hasattr(self, 'wait_counter'):
            self.wait_counter = 0
        if not hasattr(self, 'dynamic_strategies'):
            self.dynamic_strategies = []
        # Action: 0 = hold, 1 = buy, 2 = sell (close one position)
        if self.wait_counter > 0:
            self.wait_counter -= 1
            action = 0  # Force hold during waiting period
        if action == 1 and len(self.positions) < self.max_positions and self.wait_counter == 0:
            self.positions.append({'entry': self.price, 'asset_idx': self.asset_idx, 'strategy_idx': self.strategy_idx})
            self.balance -= self.transaction_cost
            reward -= self.transaction_cost
            # If now at max positions, start waiting period
            if len(self.positions) == self.max_positions:
                self.wait_counter = 10  # Wait 10 steps before allowing new trades
        elif action == 2 and self.positions:
            pos = self.positions.pop(0)
            profit = self.price - pos['entry'] - self.transaction_cost
            self.balance += profit
            if profit > 0:
                reward = profit
            elif profit < -10:
                reward = profit * 0.5
            else:
                reward = profit
            # If all positions closed, reset waiting period
            if not self.positions:
                self.wait_counter = 0
            # --- Dynamic strategy creation ---
            # If a trade is closed, record the experience as a new strategy
            new_strategy = {
                'strategy': f'RL_Strategy_{len(self.dynamic_strategies)+1}',
                'description': f'Auto-generated from trade: entry={pos["entry"]}, exit={self.price}, profit={profit:.2f}',
                'key_notes': f'Asset={self.asset}, Pattern={self.pattern_entry.get("patterns", [])}, Profit={profit:.2f}',
                'performance': profit
            }
            self.dynamic_strategies.append(new_strategy)
        else:
            if self.positions:
                unrealized = sum(self.price - pos['entry'] for pos in self.positions)
                if unrealized > 0:
                    reward += 0.1 * unrealized
        # Move to next price
        self.price_ptr += 1
        if self.price_ptr < len(self.prices):
            self.price = self.prices[self.price_ptr]
        else:
            done = True
        if self.balance <= 0:
            done = True
        state = self._get_state()
        return state, reward, done, {}
# Use a constant initial balance for all training episodes
INITIAL_BALANCE = 1000.0

env = TradingEnv(ALLOWED_ASSETS, STRATEGY_NAMES, initial_balance=INITIAL_BALANCE)

# Q-learning agent: chooses asset, strategy, and action
class QAgent:
    def __init__(self, state_size, action_size, asset_size, strategy_size, memory_size=2000):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(state_size,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(action_size * asset_size * strategy_size, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.memory = deque(maxlen=memory_size)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.98  # Faster decay
        self.action_size = action_size
        self.asset_size = asset_size
        self.strategy_size = strategy_size
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return (random.randrange(self.action_size),
                    random.randrange(self.asset_size),
                    random.randrange(self.strategy_size))
        q_values = self.model.predict(state[np.newaxis], verbose=0)[0]
        idx = np.argmax(q_values)
        action = idx % self.action_size
        asset_idx = (idx // self.action_size) % self.asset_size
        strategy_idx = idx // (self.action_size * self.asset_size)
        return action, asset_idx, strategy_idx
    def remember(self, state, action, asset_idx, strategy_idx, reward, next_state, done):
        self.memory.append((state, action, asset_idx, strategy_idx, reward, next_state, done))
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, asset_idx, strategy_idx, reward, next_state, done in minibatch:
            target = reward
            if not done:
                q_next = self.model.predict(next_state[np.newaxis], verbose=0)[0]
                target += self.gamma * np.amax(q_next)
            target_f = self.model.predict(state[np.newaxis], verbose=0)
            idx = strategy_idx * self.asset_size * self.action_size + asset_idx * self.action_size + action
            target_f[0][idx] = target
            self.model.fit(state[np.newaxis], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Update agent state_size to match new state vector (added unrealized_pnl, avg_entry)
pattern_feat_len = 3 + 3 + len(pattern_names) + 5 + 6
agent = QAgent(state_size=11+pattern_feat_len+1, action_size=3, asset_size=len(ALLOWED_ASSETS), strategy_size=len(STRATEGY_NAMES), memory_size=5000)

total_profits = []

def should_stop(profits, window=10, threshold=0.01):
    if len(profits) < window:
        return False
    recent = profits[-window:]
    avg = np.mean(recent)
    std = np.std(recent)
    # Stop if average profit is positive and std is low (stable)
    return avg > 0 and std < threshold

# Training loop with early stopping
for episode in range(1000):  # Allow up to 500 episodes
    state = env.reset()
    total_reward = 0
    for t in range(1000):
        action, asset_idx, strategy_idx = agent.act(state)
        next_state, reward, done, _ = env.step(action, asset_idx, strategy_idx)
        agent.remember(state, action, asset_idx, strategy_idx, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    agent.replay(batch_size=18)  # Reduced batch size for speed
    profit = env.balance - INITIAL_BALANCE
    total_profits.append(profit)
    print(f"Episode {episode+1}: Total Reward: {total_reward:.2f} Balance: {env.balance:.2f} Profit: {profit:.2f} Asset: {ALLOWED_ASSETS[env.asset_idx]} Strategy: {STRATEGY_NAMES[env.strategy_idx]}")
    if should_stop(total_profits):
        print(f"Early stopping at episode {episode+1} (learning curve stabilized)")
        break

final_profit = env.balance - INITIAL_BALANCE
print(f"\nTraining complete. Final balance: {env.balance:.2f} | Final profit/loss: {final_profit:.2f}")
print(f"Average profit/loss over all episodes: {np.mean(total_profits):.2f}")

# Plot learning curve
plt.figure(figsize=(10,5))
plt.plot(total_profits, label='Profit per Episode')
plt.xlabel('Episode')
plt.ylabel('Profit')
plt.title('Learning Curve: Profit per Episode')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('learning_curve.png')
plt.show()

# Optionally, expose dynamic strategies for inspection
# Example: print(env.dynamic_strategies[-5:]) after training to see the last 5 auto-generated strategies
