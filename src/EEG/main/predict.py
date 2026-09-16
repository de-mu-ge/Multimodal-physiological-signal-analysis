# Import official packages
import torch
import numpy as np
# import torch.nn as nn
from torch.utils.data import DataLoader

# Import Dataset
from src.EEG.api import KnownDataSet, UnknownDataSet
known_dataloader = DataLoader(KnownDataSet(), batch_size=1, shuffle=False, num_workers=0)
unknown_dataloader = DataLoader(UnknownDataSet(), batch_size=1, shuffle=False, num_workers=0)

# Import Config
from src.EEG.api import Config
pth_out_path = Config.Binary_emotion_classification_model_path

# Model
from src.EEG.api import EEGModel
model = EEGModel()
model.load_state_dict(torch.load(pth_out_path))
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Predict:
"""
0 : negative
1 : positive
"""
dataset_nanmes = ["known_datset", "unknown_dataset"]
def predict():
    with torch.no_grad():
        for (name, dataloader) in zip(dataset_nanmes, [known_dataloader, unknown_dataloader]):   # First the training set, then the test set.
            # Init:
            index = 0
            negative = 0
            positive = 0
            negative_true = 0
            positive_true = 0

            for i, (data, labels) in enumerate(dataloader):
                index += 1

                data, labels = data.to(device), labels.to(device)

                out = model(data)
                out = torch.sigmoid(out).cpu().detach().numpy()
                out = np.argmax(out, axis=1).tolist()[0]


                if labels == 0:
                    negative += 1
                    if out == 0:
                        negative_true += 1

                if labels == 1:
                    positive += 1
                    if out == 1:
                        positive_true += 1


            # std::Cout:
            # if index == 2500:
            #     print("Known data predict:")
            # else:
            #     print("Unknown data predict:")

            print(f"this is {name}.")
            print(f"准确率 : {(negative_true + positive_true) / index} ")
            print(f"负标签召回率 : {negative_true / negative} ")
            print(f"正标签召回 : {positive_true / positive } ")


            print("")
            print("")

if "__main__" == __name__:
    predict()




