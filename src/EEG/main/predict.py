# Import official packages
import torch
import numpy as np
import openpyxl
from pathlib import Path

# import torch.nn as nn
from torch.utils.data import DataLoader

# Import Dataset
from src.EEG.api import KnownDataSet, UnknownDataSet
known_dataloader = DataLoader(KnownDataSet(), batch_size=1, shuffle=False, num_workers=0)
unknown_dataloader = DataLoader(UnknownDataSet(), batch_size=1, shuffle=False, num_workers=0)

# Import Config
from src.EEG.api import Config
pth_out_path = Config.Binary_emotion_classification_model_path
xlsx_path = Path(__file__).parents[1] / r"data/out/Log/deap_false_data_index_json.xlsx"

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
false_data_binary_index = []
def predict() -> None:
    with torch.no_grad():
        for (name, dataloader) in zip(dataset_nanmes, [known_dataloader, unknown_dataloader]):   # First the training set, then the test set.
            # Init:
            index = 0
            negative = 0
            positive = 0
            negative_true = 0
            positive_true = 0
            false_data_index = []
            

            for i, (data, labels) in enumerate(dataloader):

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

                if labels != out:
                    false_data_index.append(index)

                index += 1

            false_data_binary_index.append(false_data_index)




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

def wb() -> None:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "按列写入数据"

    data_by_columns = false_data_binary_index 

    for col_idx, column_data in enumerate(data_by_columns, start=1):
        for row_idx, value in enumerate(column_data, start=1):
            # row 行，column 列（从1开始）
            ws.cell(row=row_idx, column=col_idx, value=value)

    # 保存文件
    wb.save(xlsx_path)


if "__main__" == __name__:
    predict() 
    wb()
    



