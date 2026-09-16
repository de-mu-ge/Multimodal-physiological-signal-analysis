from dataclasses import dataclass

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parents[3]

@dataclass
class Config:

    # Input:

    eav_dataset_path: str = r"D:\ING\create\data\eav_dataset"



    # Output:

    Binary_emotion_classification_model_path: str = r"src/Sound/data/out/pth/Binary_emotion_classification_model.pth"








        # Quantization and deployment; a separate new project will be created for it.
