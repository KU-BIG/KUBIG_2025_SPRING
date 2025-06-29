from bandit.base import Bandit
import random

class EpsilonGreedy(Bandit):
    def __init__(self):
        self.values = {}

    def update(self, item_id, reward):
        total, count = self.values.get(item_id, (0.0, 0))
        self.values[item_id] = (total + reward, count + 1)

    def get_value(self, item_id):
        total, count = self.values.get(item_id, (0.0, 0))
        return total / count if count > 0 else 0.0

    def get_score(self, item_id, epsilon=0.2):
        if random.random() < epsilon:
            score = random.uniform(0, 1)
            print(f"🎲 탐험 {item_id} | score={score:.2f}")
            return score
        else:
            total, count = self.values.get(item_id, (0.0, 0))
            score = total / count if count > 0 else 0.0
            print(f"👉 이용 {item_id} | score={score:.2f} (total={total}, count={count})")
            return score
