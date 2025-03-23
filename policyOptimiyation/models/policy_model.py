from transformers import AutoModelForSeq2SeqLM

def load_policy_model(model_path):
    """
    Load the policy model (SFT model).
    """
    return AutoModelForSeq2SeqLM.from_pretrained(model_path)