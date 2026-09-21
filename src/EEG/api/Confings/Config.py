from dataclasses import dataclass

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[4]

@dataclass
class Config:

    # Input:
    deap_data_path: str = r"D:\ING\create\data\deap_dataset"
    eav_data_path: str = r"D:\ING\create\data\eav_dataset"


    # Output:
        # This project mainly focuses on binary classification.
    Binary_emotion_classification_model_path: str = BASE_DIR / r"src/EEG/data/out/pth/Binary_emotion_classification_model.pth"

    TenClass_emotion_classification_model_path: str = BASE_DIR / r"src/EEG/data/out/pth/Ten-class_emotion_classification_model.pth"


    # Cache
        # v2：缓存里的 EegPlayer 带上了 subject / trial / source 元数据，
        #     切分依赖这些信息，所以格式和旧的 EegCache.npy 不兼容。
        #     旧缓存（无身份信息，只能按位置切）保留在磁盘上作对照，不再被读取。
    cache_version: int = 2
    cache_path: str = BASE_DIR / r"src/EEG/data/cache/EegCache_v2.npy"

    # DataSet / Split
        # 总量 47,120 条：
        #     Deap : 32 受试者 × 40 试次 × 4 段            =  5,120
        #     Eav  : 42 受试者 × 200 segment × 5 块        = 42,000
        #
        # 切分的最小不可切分单位是 (数据集, 受试者, trial)：
        #     trial = DEAP 的一个试次（4 段 2000 点）/ EAV 的一个 segment（5 块 2000 点）
        # 同一个 trial 内的片段标签相同、波形相邻，必须整组落在同一个集合，
        # 否则同一个试次的近重复片段会横跨训练集和测试集。
        #
        #     unknown = 每个数据集留出 test_subject_ratio 的受试者，整人不参与训练
        #     known   = 训练受试者中留出 known_trial_ratio 的 trial
        #     train   = 其余全部
        # 三者的样本互不相交，unknown 与 train 的受试者集合也不相交。
    split_seed: int = 42
    test_subject_ratio: float = 0.2
    known_trial_ratio: float = 0.1

    # Model
    lr : float = 0.001
    epochs : int = 100

    # Log
    log_path : str = BASE_DIR / r"src/EEG/data/out/Log/Round-by-round_inference.log"
    deap_false_data_index_json_path :str = BASE_DIR / r"src/EEG/data/out/Log/deap_false_data_index_json.json"

    pass






        # Quantization and deployment; a separate new project will be created for it.
