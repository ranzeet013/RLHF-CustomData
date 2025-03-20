import torch
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
from flanT5.preprocess import load_and_clean_data, split_data
from flanT5.dataset import create_dataset
from flanT5.model import load_model
from flanT5.evaluate import compute_metrics

# Load data
data = load_and_clean_data("/content/drive/MyDrive/RLHF/SFT/querydata.csv")
train, test = split_data(data)

# Create datasets
dataset = create_dataset(train, test)
tokenizer, model = load_model()

# Data Collator
label_pad_token_id = -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

# Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="/content/drive/MyDrive/RLHF/SFT/output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_dir="/content/drive/MyDrive/RLHF/SFT/logs",
    logging_strategy="steps",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    report_to="tensorboard",
    push_to_hub=False,
)

# Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()
