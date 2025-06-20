import numpy as np
import tensorflow as tf
import json
import random

# Load Sharia-compliant asset classes
with open('asset_screening_dataset.json') as f:
    sharia_assets = json.load(f)
ALLOWED_ASSETS = [a['asset_class'] for a in sharia_assets]

# Load trading strategies
with open('trading_strategies_dataset.json') as f:
    strategies = json.load(f)
STRATEGY_NAMES = [s['strategy'] for s in strategies]

# Simulated market environment with asset and strategy selection
class TradingEnv:
    def __init__(self, asset_classes, strategies):
        self.asset_classes = asset_classes
        self.strategies = strategies
        self.reset()
    def reset(self):
        self.balance = 1000.0
        self.asset_idx = random.randrange(len(self.asset_classes))
        self.strategy_idx = random.randrange(len(self.strategies))
        self.price = 100.0
        self.position = 0  # 0: no position, 1: long
        return self._get_state()
    def _get_state(self):
        # State includes balance, price, position, asset_idx, strategy_idx
        return np.array([self.balance, self.price, self.position, self.asset_idx, self.strategy_idx])
    def step(self, action, asset_idx, strategy_idx):
        self.asset_idx = asset_idx
        self.strategy_idx = strategy_idx
        reward = 0
        done = False
        if action == 1 and self.position == 0:
            self.position = 1
            self.entry_price = self.price
        elif action == 2 and self.position == 1:
            profit = self.price - self.entry_price
            self.balance += profit
            reward = profit
            self.position = 0
        # Simulate price change (could be improved per asset/strategy)
        self.price += np.random.randn()
        if self.balance <= 0:
            done = True
        return self._get_state(), reward, done, {}

env = TradingEnv(ALLOWED_ASSETS, STRATEGY_NAMES)

# Q-learning agent: chooses asset, strategy, and action
class QAgent:
    def __init__(self, state_size, action_size, asset_size, strategy_size):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(48, input_dim=state_size, activation='relu'),
            tf.keras.layers.Dense(48, activation='relu'),
            tf.keras.layers.Dense(action_size * asset_size * strategy_size, activation='linear')
        ])
        self.model.compile(optimizer='adam', loss='mse')
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
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
    def replay(self, batch_size=16):
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))
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

agent = QAgent(state_size=5, action_size=3, asset_size=len(ALLOWED_ASSETS), strategy_size=len(STRATEGY_NAMES))

# Training loop
for episode in range(100):
    state = env.reset()
    total_reward = 0
    for t in range(50):
        action, asset_idx, strategy_idx = agent.act(state)
        next_state, reward, done, _ = env.step(action, asset_idx, strategy_idx)
        agent.remember(state, action, asset_idx, strategy_idx, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done:
            break
    agent.replay()
    print(f"Episode {episode+1}: Total Reward: {total_reward:.2f} Balance: {env.balance:.2f} Asset: {ALLOWED_ASSETS[env.asset_idx]} Strategy: {STRATEGY_NAMES[env.strategy_idx]}")

print("Training complete. The agent has learned to select assets and strategies.")
