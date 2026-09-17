# import official package
import os
import numpy as np
import pickle   # Used to import the function for reading .dat files from the .dat library.
from scipy.io import loadmat    # Used to import the function for reading MATLAB .mat files from the SciPy library.
import random

# import person package && config
from src.EEG.api.Confings.Config import Config

cache_path = Config.cache_path      # cache path
deap_data_path = Config.deap_data_path      # deap data path
eav_data_path = Config.eav_data_path    # eav data path


# EegPlayers
class EegPlayer:
    __slots__ = ('data', 'valence')     # Avoid dictionary conversion to improve performance.
    def __init__(self, data, valence):
        self.data = data
        self.valence = valence


# DataRead
def data_read() -> list:
    play_list = []

    if os.path.exists(cache_path):

        print("read cache data")
        play_list = np.load(cache_path, allow_pickle=True)
    else:

        """
        # deap data read:
        list_dir = os.listdir(deap_data_path)
        for dirs in list_dir:
            os_dir = os.path.join(deap_data_path, dirs)
            with open(os_dir, "rb") as f:
                subject = pickle.load(f, encoding="latin1")
                label = subject['labels']
                # print(label)
                # valence = subject['valence']
                # print(label.shape)
                # break
                data = subject['data'][
                    :, :30]  # 前32个电极为脑电数据                # print(type(data))   # <class 'numpy.ndarray'>

                data = np.array(data, dtype=np.float32)

                for i in range(40):
                    # print(type(label[i]))
                    input = label[i].tolist()

                    valence = 0 if int(input[0]) <= 5 else 1

                    for j in range(4):
                        # 8064 个采样点切成 4 段互不重叠的 2016 点。
                        # 之前写成 j:j+2016，4 段只平移 1 个点，几乎完全相同，
                        # 等于把每个试次复制了 4 份，训练精度全靠背数据。
                        play_list.append(EegPlayer(data[i][:, j * 2000:(j + 1) * 2000].transpose(-1, 0), valence))

                        """
        
        # eav data read:
        data_list = []
        list_dir = os.listdir(eav_data_path)
        for dirs in list_dir:
            if dirs == "GitHub_Codes":
                continue
            # print(dirs)
            for eeg in os.listdir(os.path.join(eav_data_path, dirs, 'EEG')):
                data_list.append(os.path.join(eav_data_path, dirs, 'EEG', eeg))

        valance_list = data_list[0::2]
        label_list = data_list[1::2]

        for i in range(len(valance_list)):

            print("数据处理", i + 1, "/ 42")

            valance = loadmat(valance_list[i])
            print("数据提取")
            # 部分文件的变量名是 seg,部分是 seg1
            valance = valance['seg'] if 'seg' in valance else valance['seg1']
            valance = valance.transpose(-1, 0, 1)
            print("数据提取完毕")

            label = loadmat(label_list[i])
            label = label['label'].transpose(-1, 0)

            for _ in range(200):        # 200个样本
                # print("进入200分割循环")
                valance_data = valance[_]

                valance_data = np.array(valance_data, dtype=np.float32)

                label_data = get_negative_label(label[_])
                for j in range(5):      # 一段数据切成五分
                    play_list.append(EegPlayer(valance_data[2000 * j: 2000 * (j + 1)], label_data))


        random.shuffle(play_list)   # Shuffle the data

        play_list = np.stack(play_list)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, play_list, allow_pickle=True)


    return play_list





# functions
def get_negative_label(label) -> int:
    """
    0 = 消极
    1 = 积极

    消极：
    2 = 愤怒（听）
    3 = 愤怒（说）
    6 = 悲伤（听）
    7 = 悲伤（说）
    """
    label = np.argmax(label, axis=-1)
    if label in [2, 3, 6, 7]:
        return 0    # 消极
    else:
        return 1    # 积极


if __name__ == "__main__":
    play_list = data_read()
    
    pass
    # play_list = data_read()
    # print(len(play_list))       # 47120
