import os
import torch
from transformers import AutoTokenizer, EsmForProteinFolding

MODEL_NAME = "/home/ubuntu/esm_weights"

def test_load():
    print(f"Loading model from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = EsmForProteinFolding.from_pretrained(MODEL_NAME, weights_only=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    print(f"Model loaded successfully on {device}")

    sequence = "GWSTELEKHREELKEFLKKEGITNVEIRIDNGRLEVRVEGGTERLKRFLEELRQKLEKKGYTVDIKIE"
    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    plddt = outputs.plddt[0].cpu().numpy()
    print(f"pLDDT: {plddt.mean():.4f}")

if __name__ == "__main__":
    test_load()

