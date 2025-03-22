from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from data.dataset import PreferenceDataset
from utils.tokenizer import collate_fn
from utils.evaluation import evaluate_reward_model
from config import Config
import torch

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_CHECKPOINT, num_labels=1)
model.to(Config.DEVICE)

# Load dataset
dataset = PreferenceDataset(Config.DATA_PATH, tokenizer, Config.MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

# Evaluate the model
accuracy, spearman_corr = evaluate_reward_model(model, dataloader, Config.DEVICE)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Spearman's Rank Correlation: {spearman_corr:.4f}")
