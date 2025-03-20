import torch
from scripts.model import load_model

tokenizer, model = load_model()

def generate_response(prompt):
    inputs = tokenizer("summarize: " + prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
    output = model.generate(**inputs)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    sample_prompt = "What is the capital of France?"
    response = generate_response(sample_prompt)
    print(f"Model Response: {response}")
