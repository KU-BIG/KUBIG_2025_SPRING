# bandit/thompson_sampling.py
from bandit.base import Bandit
import random

class ThompsonSampling(Bandit):
    def __init__(self):
        self.rewards = {}

    def update(self, item_id, reward):
        total_reward, total_count = self.rewards.get(item_id, (0.0, 0))
        total_reward += reward  # reward ∈ [0.0, 1.0]
        total_count += 1
        self.rewards[item_id] = (total_reward, total_count)

    def get_value(self, item_id):
        total_reward, total_count = self.rewards.get(item_id, (0.0, 1))
        return total_reward / total_count

    def get_score(self, item_id):
        total_reward, total_count = self.rewards.get(item_id, (0.0, 1))
        alpha = 1.0 + total_reward
        beta = 1.0 + total_count - total_reward
        score = random.betavariate(alpha, beta)
        print(f"🎯 탐험/이용 {item_id} | Beta({alpha:.1f},{beta:.1f}) → score={score:.2f}")
        return score