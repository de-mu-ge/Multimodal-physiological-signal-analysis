# Import official packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

# Import Dataset
from src.EEG.api import TrainDataSet
train_dataloader = DataLoader(TrainDataSet(),batch_size=32,shuffle=True, num_workers=0)

from src.EEG.api import KnownDataSet
known_dataloader = DataLoader(KnownDataSet(), batch_size=1,shuffle=False, num_workers=0)
from src.EEG.api import UnknownDataSet
unknown_dataloader = DataLoader(UnknownDataSet(), batch_size=1,shuffle=False, num_workers=0)

# Import Config
from src.EEG.api import Config
lr = Config.lr
epochs = Config.epochs
log_path = Config.log_path

# Logs
import logging
logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)

# Model
from src.EEG.api import EEGModel
model = EEGModel()
model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.to(device)

# Round-by-round_inference
def round_by_round_inference():
    for epoch in range(epochs):

        # this is train.
        print(f"this is {epoch + 1} epoch")     # print epochs
        logging.info(f"this is {epoch + 1} epoch")

        loss_num = 0
        i = 1

        for batch_idx, (data, target) in enumerate(train_dataloader):

            i = i + 1

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_num += loss.item()
        print("loss.item(): ", loss_num / i)
        logging.info(f"loss.item(): {loss_num / i}")


        # this is preict.
        print("")
        dataset_nanmes = ["known_datset", "unknown_dataset"]
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

                logging.info(f"this is {name}.")
                logging.info(f"准确率 : {(negative_true + positive_true) / index} ")
                logging.info(f"负标签召回率 : {negative_true / negative} ")
                logging.info(f"正标签召回 : {positive_true / positive } ")

    
                print("")
                print("")


    

if __name__ == "__main__":
    logging.info("start round_by_round_inference():")
    round_by_round_inference()




