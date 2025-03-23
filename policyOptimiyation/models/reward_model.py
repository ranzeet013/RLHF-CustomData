from transformers import AutoModelForSeq2SeqLM

def load_reward_model(model_path):
    """
    Load the reward model.
    """
    reward_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    reward_model.eval()  
    return reward_model