import numpy as np
from sapsan.core.models import Dataset


class FakeDataset(Dataset):
    """Fake dataset for demo mocks"""
    def __init__(self,
                 size: int = 1000,
                 n_features: int = 4,
                 seed: int = 42):
        self.size = size
        self.n_features = n_features
        self.seed = seed
        np.random.seed(self.seed)

    def load(self):
        x = np.random.random((self.size, self.n_features))
        y = np.random.random(self.size)
        return x, y
