import torch

from metrics.base_metric import BaseMetric


class Accuracy(BaseMetric):

    def __init__(self, name='Accuracy'):
        super().__init__(name)

    def compute_metric(self):
        accuracy = self._num_corrects / self._num_samples
        return accuracy
