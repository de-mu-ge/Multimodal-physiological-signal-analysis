import torch
from torch.utils.data import Dataset
from src.EEG.api.data_read.data_read import data_read

# unknown_datas = play_list[45000:]

_unknown_datas = None

def get_unknown_datas():
    global _unknown_datas
    if _unknown_datas is None:
        _unknown_datas = data_read()[45000:]
    return _unknown_datas

class UnknownDataSet(Dataset):      # unknown dataset
    def __init__(self, ):
        self.data = [obj.data for obj in get_unknown_datas()]
        self.labels = [obj.valence for obj in get_unknown_datas()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), self.labels[idx]