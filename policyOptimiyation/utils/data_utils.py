from transformers import AutoTokenizer

def load_tokenizer(model_path):
    """
    Load the tokenizer.
    """
    return AutoTokenizer.from_pretrained(model_path)

def tokenize_text(tokenizer, text):
    """
    Tokenize input text.
    """
    return tokenizer(text, return_tensors="pt").input_ids