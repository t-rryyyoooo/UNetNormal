import torch.utils.data as data
from .utils import separateData
from pathlib import Path

class UNetDataset(data.Dataset):
    def __init__(self, dataset_path=None, phase="train", criteria=None, transform=None):
        self.transform = transform
        self.phase = phase

        self.data_list = separateData(dataset_path, criteria, phase)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        path = self.data_list[index]
        imageArray, labelArray = self.transform(self.phase, *path)

        return imageArray, labelArray


