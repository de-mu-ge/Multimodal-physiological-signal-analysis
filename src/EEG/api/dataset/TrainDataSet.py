import torch
from torch.utils.data import Dataset
from src.EEG.api.data_read.data_read import data_read
from src.EEG.api.Confings.Config import Config
num = Config.trains_num

# train_datas = play_list[:num]     #  Instantiate the function

_train_datas = None

def get_train_datas():
    global _train_datas
    if _train_datas is None:
        _train_datas = data_read()[:num]
    return _train_datas

class TrainDataSet(Dataset):        # train dataset
    def __init__(self,):
        self.data = [obj.data for obj in get_train_datas()]
        self.labels = [obj.valence for obj in get_train_datas()]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float(), self.labels[idx]


