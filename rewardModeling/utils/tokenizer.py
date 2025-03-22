from torch.nn.utils.rnn import pad_sequence
import torch

def collate_fn(batch, pad_token_id):
    # Extract inputs and labels
    preferred_input_ids = [item["preferred_input_ids"] for item in batch]
    preferred_attention_mask = [item["preferred_attention_mask"] for item in batch]
    non_preferred_input_ids = [item["non_preferred_input_ids"] for item in batch]
    non_preferred_attention_mask = [item["non_preferred_attention_mask"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32)

    # Pad sequences to the same length within the batch
    preferred_input_ids = pad_sequence(preferred_input_ids, batch_first=True, padding_value=pad_token_id)
    preferred_attention_mask = pad_sequence(preferred_attention_mask, batch_first=True, padding_value=0)
    non_preferred_input_ids = pad_sequence(non_preferred_input_ids, batch_first=True, padding_value=pad_token_id)
    non_preferred_attention_mask = pad_sequence(non_preferred_attention_mask, batch_first=True, padding_value=0)

    return {
        "preferred_input_ids": preferred_input_ids,
        "preferred_attention_mask": preferred_attention_mask,
        "non_preferred_input_ids": non_preferred_input_ids,
        "non_preferred_attention_mask": non_preferred_attention_mask,
        "label": labels
    }
