import pandas as pd
from torch.utils.data import Dataset
from utils import ROOT_PATH

class BurnoutDataset(Dataset):
    def __init__(self, data_type):
        if data_type == "train":
            df = pd.read_csv(ROOT_PATH / "data" / "train_data.csv")
        elif data_type == "test":
            df = pd.read_csv(ROOT_PATH / "data" / "test_data.csv")
        elif data_type == "val":
            df = pd.read_csv(ROOT_PATH / "data" / "val_data.csv")
        else:
            raise NotImplementedError

        self.x = df.drop('target', axis=1).values
        self.y = df['target'].values

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class ModelHistory:
    def __init__(self, **kwargs):
        self.history = {
            'loss': [],
        }
        self.metrics = dict()
        for k, v in kwargs.items():
            self.metrics[k] = v

    def calculate_metrics(self, y_pred, y_test):
        for k, v in self.metrics.items():
            self.history[k].append(v(y_pred, y_test))

    def get_metrics(self):
        return ['loss'].extend(list(self.metrics.keys()))

    def add_loss(self, loss):
        self.history['loss'].append(loss)
