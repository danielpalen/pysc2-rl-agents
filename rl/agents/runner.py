from abc import ABC, abstractmethod

class BaseRunner(ABC):

    @abstractmethod
    def run_batch(self, train_summary):
        pass

    @abstractmethod
    def get_mean_score(self):
        pass
