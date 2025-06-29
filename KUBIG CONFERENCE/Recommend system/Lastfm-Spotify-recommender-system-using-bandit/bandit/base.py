from abc import ABC, abstractmethod

class Bandit(ABC):
    @abstractmethod
    def update(self, item_id: str, reward: float):
        pass
