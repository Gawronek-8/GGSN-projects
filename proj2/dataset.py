from torch.utils.data import Dataset
from torch import nn

class FireDataset(Dataset):
    def __init__(self, train=True, transforms=None, transform_p = 0.3):
        self.transforms = transforms or nn.Identity()
        self.transform_p = transform_p

        if train:
            

    def __len__(self):
