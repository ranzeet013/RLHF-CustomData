import torch

def ppo_loss(old_probs, new_probs, advantages, clip_epsilon=0.2):
    """
    Compute the PPO loss.
    """
    ratio = new_probs / old_probs
    clipped_ratio = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    return -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

def compute_advantages(rewards, gamma=0.99):
    """
    Compute advantages using discounted cumulative rewards.
    """
    advantages = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        advantages.insert(0, R)
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return advantages