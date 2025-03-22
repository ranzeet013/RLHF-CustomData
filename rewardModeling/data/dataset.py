# data/dataset.py

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PreferenceDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path)

    def _load_data(self, data_path):
        df = pd.read_csv(data_path)
        df["label"] = df["preference"].apply(lambda x: 1 if x == "human" else 0)
        df = df.drop(columns=["Unnamed: 0"], axis=1)
        return df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prompt = row["prompt"]
        human_answer = row["human_answer"]
        machine_answer = row["machine_answer"]
        label = row["label"]

        # Tokenize the preferred and non-preferred outputs
        preferred = self.tokenizer(
            prompt + human_answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        non_preferred = self.tokenizer(
            prompt + machine_answer,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )

        return {
            "preferred_input_ids": preferred["input_ids"].squeeze(0),
            "preferred_attention_mask": preferred["attention_mask"].squeeze(0),
            "non_preferred_input_ids": non_preferred["input_ids"].squeeze(0),
            "non_preferred_attention_mask": non_preferred["attention_mask"].squeeze(0),
            "label": label
        }