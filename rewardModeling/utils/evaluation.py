from sklearn.metrics import accuracy_score
from scipy.stats import spearmanr
import numpy as np
import torch 

def evaluate_reward_model(reward_model, dataloader, device):
    reward_model.eval()
    all_labels = []
    all_predictions = []
    all_preferred_rewards = []
    all_non_preferred_rewards = []

    with torch.no_grad():
        for batch in dataloader:
            # Move tensors to the correct device
            preferred_input_ids = batch["preferred_input_ids"].to(device)
            preferred_attention_mask = batch["preferred_attention_mask"].to(device)
            non_preferred_input_ids = batch["non_preferred_input_ids"].to(device)
            non_preferred_attention_mask = batch["non_preferred_attention_mask"].to(device)
            labels = batch["label"].to(device)

            # Get rewards
            preferred_rewards = reward_model(input_ids=preferred_input_ids, attention_mask=preferred_attention_mask).logits
            non_preferred_rewards = reward_model(input_ids=non_preferred_input_ids, attention_mask=non_preferred_attention_mask).logits

            # Predict preferences
            predictions = (preferred_rewards > non_preferred_rewards).int().squeeze()

            # Store results
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_preferred_rewards.extend(preferred_rewards.cpu().numpy())
            all_non_preferred_rewards.extend(non_preferred_rewards.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions)
    reward_differences = np.array(all_preferred_rewards) - np.array(all_non_preferred_rewards)
    spearman_corr, _ = spearmanr(reward_differences, all_labels)

    return accuracy, spearman_corr
