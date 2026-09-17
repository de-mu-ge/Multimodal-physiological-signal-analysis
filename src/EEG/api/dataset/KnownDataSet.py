import torch
from torch.utils.data import Dataset
from src.EEG.api.data_read.data_read import data_read
from src.EEG.api.Confings.Config import Config
num = Config.test_known_num

# known_datas = play_list[:num]

_known_datas = None

def get_known_datas():
    global _known_datas
    if _known_datas is None:
        _known_datas = data_read()[:num]
    return _known_datas


class KnownDataSet(Dataset):        # known dataset
    def __init__(self, ):
        self.data = [obj.data for obj in get_known_datas()]
        self.labels = [obj.valence for obj in get_known_datas()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), self.labels[idx]