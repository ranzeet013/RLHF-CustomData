import torch

def compute_advantages(rewards, gamma=0.99):
    """
    Compute advantages using discounted cumulative rewards.

    Args:
        rewards (list): List of rewards for each timestep.
        gamma (float): Discount factor.

    Returns:
        torch.Tensor: Tensor of advantages.
    """
    advantages = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        advantages.insert(0, R)
    advantages = torch.tensor(advantages)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # Normalize advantages
    return advantages