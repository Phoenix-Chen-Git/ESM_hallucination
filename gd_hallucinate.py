import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import time
import json
from pathlib import Path
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils import make_atom14_masks
from transformers.models.esm.modeling_esmfold import categorical_lddt
from tqdm import tqdm

# Constants
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

class GDHallucinator:
    def __init__(self, model_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = EsmForProteinFolding.from_pretrained(model_path, weights_only=False)
        self.model.to(self.device)
        self.model.eval()
        
        # Monkey-patch EsmEmbeddings.forward to fix a bug in transformers when using inputs_embeds
        # where it tries to use input_ids even if it's None.
        def patched_forward(self_emb, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
            if inputs_embeds is None:
                inputs_embeds = self_emb.word_embeddings(input_ids)
            
            if position_ids is None:
                if input_ids is not None:
                    position_ids = torch.arange(input_ids.size(1), device=input_ids.device).expand_as(input_ids)
                else:
                    position_ids = torch.arange(inputs_embeds.size(1), device=inputs_embeds.device).unsqueeze(0)
            
            position_embeddings = self_emb.position_embeddings(position_ids)
            embeddings = inputs_embeds + position_embeddings
            if self_emb.layer_norm is not None:
                embeddings = self_emb.layer_norm(embeddings)
            if self_emb.dropout is not None:
                embeddings = self_emb.dropout(embeddings)
            
            if input_ids is not None:
                embeddings = embeddings.masked_fill((input_ids == self_emb.mask_token_id).unsqueeze(-1), 0.0)
            
            return embeddings

        # Apply patch to all ESM embedding layers (there's usually only one)
        self.model.esm.embeddings.forward = patched_forward.__get__(self.model.esm.embeddings, self.model.esm.embeddings.__class__)

        # Freeze all model parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Map AA_ALPHABET to model's internal indices (0-19 for standard AAs)
        # Note: ESMFold internal tokenizer maps AAs to 0-20. 
        # We'll use the tokenizer to ensure correct mapping.
        self.aa_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(aa) for aa in AA_ALPHABET], device=self.device)
        
        # ESM-2 special tokens
        self.esm_cls_id = self.model.esm_dict_cls_idx
        self.esm_eos_id = self.model.esm_dict_eos_idx
        self.esm_pad_id = self.model.esm_dict_padding_idx
        
        # Precompute mapping for ESM backbone
        # self.model.af2_to_esm maps 0-20 to ESM-2 tokens
        self.esm_aa_ids = self.model.af2_to_esm[self.aa_ids + 1]

    def forward_soft(self, logits, num_recycles=None):
        """
        Differentiable forward pass through ESMFold using soft embeddings.
        logits: (B, L, 20)
        """
        B, L, _ = logits.shape
        probs = F.softmax(logits, dim=-1)
        
        # 1. ESM-2 backbone soft embeddings
        esm_weight = self.model.esm.embeddings.word_embeddings.weight
        probs = probs.to(esm_weight.dtype)
        soft_esm_emb = torch.matmul(probs, esm_weight[self.esm_aa_ids]) # (B, L, D_esm)
        
        cls_emb = esm_weight[self.esm_cls_id].view(1, 1, -1).expand(B, 1, -1)
        eos_emb = esm_weight[self.esm_eos_id].view(1, 1, -1).expand(B, 1, -1)
        
        inputs_embeds_esm = torch.cat([cls_emb, soft_esm_emb, eos_emb], dim=1)
        
        # ESM forward (Allow gradients through activations)
        esm_outputs = self.model.esm(inputs_embeds=inputs_embeds_esm, output_hidden_states=True)
        esm_s = torch.stack(esm_outputs.hidden_states, dim=2) # (B, L+2, nLayers, D_esm)
        esm_s = esm_s[:, 1:-1] # (B, L, nLayers, D_esm)
        
        # 2. Projection
        esm_s = esm_s.to(self.model.esm_s_combine.dtype)
        # Note: self.model.esm_s_combine.softmax(0) weights the layers
        esm_s = (self.model.esm_s_combine.softmax(0).unsqueeze(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.model.esm_s_mlp(esm_s)
        
        # 3. Trunk soft embeddings
        trunk_weight = self.model.embedding.weight
        soft_trunk_emb = torch.matmul(probs.to(trunk_weight.dtype), trunk_weight[self.aa_ids])
        s_s_0 += soft_trunk_emb
        
        # 4. Trunk forward
        # Use argmax for discrete inputs to the trunk (non-differentiable parts like relative pos)
        aa_input = self.aa_ids[torch.argmax(logits, dim=-1)]
        
        attention_mask = torch.ones((B, L), device=self.device)
        position_ids = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        s_z_0 = s_s_0.new_zeros(B, L, L, self.model.config.esmfold_config.trunk.pairwise_state_dim)
        
        # Call trunk
        structure = self.model.trunk(s_s_0, s_z_0, aa_input, position_ids, attention_mask, no_recycles=num_recycles)
        
        # 5. Add required keys for output_to_pdb
        structure["aatype"] = aa_input
        structure["residue_index"] = position_ids
        
        # Use make_atom14_masks to generate the atom masks
        make_atom14_masks(structure)

        # 6. Prediction heads
        disto_logits = self.model.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits

        lm_logits = self.model.lm_head(structure["s_s"])
        structure["lm_logits"] = lm_logits
        
        # Compute pLDDT from states using the lddt_head
        # states shape: (num_recycles+1, B, L, hidden_dim)
        lddt_head_output = self.model.lddt_head(structure["states"])  # (num_recycles+1, B, L, 37, bins)
        lddt_head_output = lddt_head_output.reshape(
            structure["states"].shape[0], B, L, -1, self.model.lddt_bins
        )
        structure["lddt_head"] = lddt_head_output
        
        # Use the last recycle's predictions
        plddt = categorical_lddt(lddt_head_output[-1], bins=self.model.lddt_bins)  # (B, L, 37)
        structure["plddt"] = plddt
            
        return structure

    def optimize(self, length, steps=100, lr=0.1, out_dir="gd_results", seed_seq=None, fixed_pos=None):
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize logits
        if seed_seq:
            # Convert sequence to initial logits
            init_logits = torch.zeros(1, len(seed_seq), 20, device=self.device)
            for i, aa in enumerate(seed_seq):
                if aa in AA_ALPHABET:
                    idx = AA_ALPHABET.index(aa)
                    init_logits[0, i, idx] = 5.0 # Give it some weight
            logits = init_logits.detach().requires_grad_(True)
            length = len(seed_seq)
        else:
            logits = (torch.randn(1, length, 20, device=self.device) * 0.01).detach().requires_grad_(True)
            
        optimizer = torch.optim.Adam([logits], lr=lr)
        
        # Fixed positions mask
        fixed_mask = torch.zeros(length, dtype=torch.bool, device=self.device)
        if fixed_pos:
            for p in fixed_pos.split(","):
                if "-" in p:
                    start, end = map(int, p.split("-"))
                    fixed_mask[start-1:end] = True
                else:
                    fixed_mask[int(p)-1] = True

        history = []
        best_plddt = 0.0
        pbar = tqdm(range(steps))
        
        for i in pbar:
            optimizer.zero_grad()
            
            structure = self.forward_soft(logits)
            
            # Loss: Maximize mean pLDDT
            plddt = structure["plddt"] # (B, L, 37)
            loss = -plddt.mean() # We want to maximize pLDDT
            
            loss.backward()
            
            # Mask gradients for fixed positions
            if fixed_pos:
                logits.grad[:, fixed_mask, :] = 0
                
            optimizer.step()
            
            # Logging
            current_plddt = -loss.item()
            if current_plddt > best_plddt:
                best_plddt = current_plddt
            pbar.set_description(f"pLDDT: {current_plddt:.4f}")
            
            if i % 10 == 0 or i == steps - 1:
                # Save best sequence and PDB
                current_seq = "".join([AA_ALPHABET[idx] for idx in torch.argmax(logits[0], dim=-1).cpu().numpy()])
                
                with torch.no_grad():
                    pdb_str = self.model.output_to_pdb(structure)[0]
                
                with open(out_path / f"step_{i:04d}.pdb", "w") as f:
                    f.write(pdb_str)
                
                history.append({
                    "step": i,
                    "score": current_plddt,
                    "best_score": best_plddt,
                    "seq": current_seq
                })

        # Save final
        final_seq = "".join([AA_ALPHABET[idx] for idx in torch.argmax(logits[0], dim=-1).cpu().numpy()])
        with open(out_path / "final_best.pdb", "w") as f:
            f.write(pdb_str)
        with open(out_path / "best_seq.txt", "w") as f:
            f.write(f">hallucinated_plddt_{current_plddt:.4f}\n{final_seq}\n")
        
        # Save history
        with open(out_path / "history.json", "w") as f:
            json.dump(history, f, indent=4)
            
        print(f"Final pLDDT: {current_plddt:.4f}")
        print(f"History saved to {out_path / 'history.json'}")
        return final_seq

def main():
    parser = argparse.ArgumentParser(description="ESMFold Hallucination (Gradient Descent)")
    parser.add_argument("--length", type=int, default=100, help="Length of sequence")
    parser.add_argument("--steps", type=int, default=100, help="Number of GD steps")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--out_dir", type=str, default="gd_hallucination", help="Output directory")
    parser.add_argument("--weights", type=str, default="/home/ubuntu/FORD/esmfold/esm_weights", help="Model weights")
    parser.add_argument("--seed_seq", type=str, default=None, help="Starting sequence")
    parser.add_argument("--fixed_pos", type=str, default=None, help="Fixed positions (1-indexed)")
    
    args = parser.parse_args()
    
    hallucinator = GDHallucinator(args.weights)
    hallucinator.optimize(
        length=args.length,
        steps=args.steps,
        lr=args.lr,
        out_dir=args.out_dir,
        seed_seq=args.seed_seq,
        fixed_pos=args.fixed_pos
    )

if __name__ == "__main__":
    main()

