import torch

import numpy as np
from sklearn.metrics import ndcg_score


class Average:
    def __init__(self):
        self.sum = 0.
        self.total = 0.

    def update(self, val, weight=1):
        self.sum += val
        self.total += weight

    def compute(self):
        avg = self.sum / self.total
        self.sum = 0.
        self.total = 0.
        return avg


class NDCG:
    def __init__(self, kind="exponential", k=10):
        self.kind = kind
        self.k = k

        self.ndcg_sum = 0.
        self.total = 0.

    def update(self, score, target, length):
        cum_length = np.cumsum(length.cpu().numpy())
        score_per_list = np.split(score.cpu().numpy(), cum_length[:-1])
        target_per_list = np.split(target.cpu().numpy(), cum_length[:-1])

        for score_of_list, target_of_list in zip(score_per_list, target_per_list):
            if self._should_count_list(target_of_list):
                gain_of_list = self._compute_gain(target_of_list)
                self.ndcg_sum += ndcg_score([gain_of_list], [score_of_list], k=self.k)
                self.total += 1

    def _compute_gain(self, target):
        if self.kind == "exponential":
            return np.power(2, target) - 1
        elif self.kind == "linear":
            return target
        else:
            raise ValueError(f"kind={self.kind} is not supported")

    def _should_count_list(self, target):
        # If the list is constant, don't count it.
        if np.all(target == target[0]):
            return False
        return True

    def compute(self):
        if self.total == 0:
            return torch.nan

        agg = self.ndcg_sum / self.total
        self.ndcg_sum = 0.
        self.total = 0.
        return agg


class TopNDCG(NDCG):
    def __init__(self, max_target=None, **kwargs):
        super().__init__(**kwargs)
        self.max_target = max_target

    def _should_count_list(self, target):
        if target.max() < self.max_target:
            return False
        return super()._should_count_list(target)
