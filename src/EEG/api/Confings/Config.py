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
    cache_path: str = BASE_DIR / r"src/EEG/data/cache/EegCache.npy"

    # Model
    lr : float = 0.001
    epochs : int = 40

    pass






        # Quantization and deployment; a separate new project will be created for it.
