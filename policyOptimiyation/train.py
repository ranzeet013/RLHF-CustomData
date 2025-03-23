import torch
from torch.optim import Adam
from config import LEARNING_RATE, EPOCHS, SFT_MODEL_PATH, REWARD_MODEL_PATH, OPTIMIZED_MODEL_PATH
from models.policy_model import load_policy_model
from models.reward_model import load_reward_model
from utils.data_utils import load_tokenizer
from utils.ppo_utils import ppo_loss
from generate_trajectories import generate_trajectories
from utils.compute_advantages import compute_advantages

# Load models and tokenizer
policy_model = load_policy_model(SFT_MODEL_PATH)
reward_model = load_reward_model(REWARD_MODEL_PATH)
tokenizer = load_tokenizer(SFT_MODEL_PATH)

# Optimizer
optimizer = Adam(policy_model.parameters(), lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    
    prompts = [
        "What is the capital of France?",
        "Explain the concept of gravity.",
        "Write a short story about a robot.",
    ]
    trajectories = generate_trajectories(policy_model, reward_model, tokenizer, prompts)

    states, actions, rewards = zip(*trajectories)
    rewards = torch.tensor(rewards)
    
    advantages = compute_advantages(rewards)

    for state, action, advantage, reward in zip(states, actions, advantages, rewards):
        optimizer.zero_grad()
        
        with torch.no_grad():
            old_logits = policy_model(input_ids=state, labels=action).logits
            old_probs = torch.softmax(old_logits, dim=-1)
            old_action_prob = old_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        new_logits = policy_model(input_ids=state, labels=action).logits
        new_probs = torch.softmax(new_logits, dim=-1)
        new_action_prob = new_probs.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        loss = ppo_loss(old_action_prob, new_action_prob, advantage)
        
        loss.backward()
        optimizer.step()
    
        print(f"Loss: {loss.item()}, Reward: {reward.item()}, Advantage: {advantage.item()}")

# Save the optimized policy model
policy_model.save_pretrained(OPTIMIZED_MODEL_PATH)