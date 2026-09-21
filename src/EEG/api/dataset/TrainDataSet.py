import torch
from torch.utils.data import Dataset

from src.EEG.api.dataset.split import get_splits
from src.EEG.api.data_read.data_read import data_read


class TrainDataSet(Dataset):        # train dataset
    def __init__(self):
        play_list = data_read()
        index = get_splits()["train"]
        self.data = [play_list[i].data for i in index]
        self.labels = [play_list[i].valence for i in index]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), self.labels[idx]
