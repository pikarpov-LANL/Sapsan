from sapsan.general.models import Dataset


class JHTDBDataset(Dataset):
    def __init__(self, path):
        self.path = path

    def load(self):
        pass