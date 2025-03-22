from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from data.dataset import PreferenceDataset
from utils.tokenizer import collate_fn
from models.reward_model import RewardModelTrainer
from config import Config
import torch
from transformers import AdamW

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSequenceClassification.from_pretrained(Config.MODEL_CHECKPOINT, num_labels=1)
model.to(Config.DEVICE)

# Load dataset
dataset = PreferenceDataset(Config.DATA_PATH, tokenizer, Config.MAX_LENGTH)
dataloader = DataLoader(dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=lambda x: collate_fn(x, tokenizer.pad_token_id))

# Initialize trainer
optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
loss_fn = torch.nn.BCEWithLogitsLoss()
trainer = RewardModelTrainer(model, Config.DEVICE, optimizer, loss_fn)

# Train the model
trainer.train(dataloader, Config.EPOCHS)
