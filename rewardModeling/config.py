import torch

class Config:
    # Paths
    DATA_PATH = "/content/drive/MyDrive/rlhgsumma/results.csv"
    MODEL_CHECKPOINT = "/content/drive/MyDrive/rlhgsumma/checkpoint-138"
    OUTPUT_DIR = "/content/drive/MyDrive/rlhgsumma/output"

    # Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-5
    EPOCHS = 3
    MAX_LENGTH = 512

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"