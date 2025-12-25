import torch
from transformers import AutoTokenizer, EsmForProteinFolding

model = EsmForProteinFolding.from_pretrained('/home/ubuntu/FORD/esmfold/esm_weights', weights_only=False).float()
tokenizer = AutoTokenizer.from_pretrained('/home/ubuntu/FORD/esmfold/esm_weights')
inputs = tokenizer(['MAGAMAGAMAGA'], return_tensors='pt', add_special_tokens=False)

model.eval()
with torch.no_grad():
    outputs = model(**inputs)

print('output keys:', outputs.keys())
print('plddt shape:', outputs.plddt.shape)
print('plddt range:', outputs.plddt.min().item(), outputs.plddt.max().item())

