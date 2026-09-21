# Import official packages
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import Dataset
from src.EEG.api import TrainDataSet, describe
dataloader = DataLoader(TrainDataSet(),batch_size=32,shuffle=True, num_workers=0)

# Import Config
from src.EEG.api import Config
lr = Config.lr
epochs = Config.epochs
pth_out_path = Config.Binary_emotion_classification_model_path

# Model
from src.EEG.api import EEGModel
model = EEGModel()
model.train()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
model.to(device)

# Train
def train():
    print(describe())

    for epoch in range(epochs):

        print(f"this is {epoch + 1} epoch")     # print epochs

        loss_num = 0
        i = 1

        for batch_idx, (data, target) in enumerate(dataloader):

            i = i + 1

            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_num += loss.item()
        print("loss.item(): ", loss_num / i)


    torch.save(model.state_dict(), pth_out_path)
    print(f"model saved in {pth_out_path}")

if __name__ == "__main__":
    train()




