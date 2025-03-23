import torch
from transformers import AutoTokenizer
from models.policy_model import load_policy_model
from models.reward_model import load_reward_model
from utils.data_utils import load_tokenizer, tokenize_text

def generate_trajectories(policy_model, reward_model, tokenizer, prompts):
    """
    Generate trajectories using the policy model and compute rewards using the reward model.
    """
    trajectories = []
    for prompt in prompts:
        input_ids = tokenize_text(tokenizer, prompt)
        with torch.no_grad():
            outputs = policy_model.generate(input_ids, max_length=50)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Get reward from reward model
        with torch.no_grad():
            reward_input_ids = tokenize_text(tokenizer, generated_text)
            reward = reward_model(reward_input_ids).logits.mean().item()
        
        trajectories.append((input_ids, outputs, reward))
    return trajectories