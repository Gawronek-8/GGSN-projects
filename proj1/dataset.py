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
    """
    Simple class for tracking model's metrics, for each different evaluation add new step
    """

    # TODO switch steps to 3 static names: training, val, test
    def __init__(self, steps, **kwargs):
        self.history = dict()
        self.metrics = dict()

        for step in steps:
            self.history[step] = {
                'loss': [],
            }

        for k, v in kwargs.items():
            self.metrics[k] = v
            for step in steps:
                self.history[step][k] = []


    def calculate_metrics(self, step_name, y_true, y_pred):
        for k, metric in self.metrics.items():
            self.history[step_name][k].append(metric(y_true, y_pred))


    def get_metrics(self):
        return ['loss'].extend(list(self.metrics.keys()))

    def add_loss(self, step_name, loss):
        self.history[step_name]['loss'].append(loss)
