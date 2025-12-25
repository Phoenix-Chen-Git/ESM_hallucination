#!/usr/bin/env python3
"""
Linker Design Script
Design a rigid linker between two protein chains using ESMFold.
Optimizes for: (1) High pLDDT of linker (2) High TM-score to ground truth
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils import make_atom14_masks
from transformers.models.esm.modeling_esmfold import categorical_lddt
from tqdm import tqdm
from Bio.PDB import PDBParser
import tmtools

# Constants
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_3TO1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}


def parse_pdb(pdb_path, chain_id, min_residue=None, max_residue=None):
    """Extract sequence and CA coordinates from PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    
    sequence = []
    ca_coords = []
    
    for model in structure:
        for chain in model:
            if chain.id == chain_id:
                for residue in chain:
                    res_id = residue.id[1]
                    if min_residue and res_id < min_residue:
                        continue
                    if max_residue and res_id > max_residue:
                        break
                    res_name = residue.get_resname()
                    if res_name in AA_3TO1:
                        sequence.append(AA_3TO1[res_name])
                        if 'CA' in residue:
                            ca_coords.append(residue['CA'].get_coord())
                        else:
                            ca_coords.append(np.array([0.0, 0.0, 0.0]))
                break  # Only first model
        break
    
    return "".join(sequence), np.array(ca_coords)


def compute_tm_score(coords1, coords2, seq1, seq2):
    """Compute TM-score between two structures using tmtools."""
    # Align based on shorter length
    min_len = min(len(coords1), len(coords2))
    coords1 = coords1[:min_len]
    coords2 = coords2[:min_len]
    
    result = tmtools.tm_align(coords1, coords2, seq1[:min_len], seq2[:min_len])
    return result.tm_norm_chain1  # TM-score normalized by first chain


class LinkerDesigner:
    def __init__(self, model_path, device=None, multi_gpu=False, gradient_checkpointing=False):
        self.multi_gpu = multi_gpu
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = EsmForProteinFolding.from_pretrained(model_path, weights_only=False)
        self.model.to(self.device)
        
        if gradient_checkpointing:
            print("Enabling gradient checkpointing for memory efficiency...")
            self.model.esm.encoder.gradient_checkpointing = True
        
        # Set trunk chunk_size to reduce memory for structure module
        if hasattr(self.model, 'trunk') and hasattr(self.model.trunk, 'chunk_size'):
            self.model.trunk.chunk_size = 32  # Process in smaller chunks
            print(f"Set trunk chunk_size to 32 for memory efficiency")
        
        # Reduce number of recycles for memory efficiency
        if hasattr(self.model, 'trunk') and hasattr(self.model.trunk, 'num_recycles'):
            self.model.trunk.num_recycles = 2  # Reduce from default 4
            print(f"Set num_recycles to 2 for memory efficiency")
        
        self.model.eval()
        
        # Monkey-patch for inputs_embeds support
        def patched_forward(self_emb, input_ids=None, attention_mask=None, position_ids=None, 
                           inputs_embeds=None, past_key_values_length=0):
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
        
        self.model.esm.embeddings.forward = patched_forward.__get__(
            self.model.esm.embeddings, self.model.esm.embeddings.__class__
        )
        
        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False
        
        # AA mappings
        self.embed_device = self.device
        self.aa_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(aa) for aa in AA_ALPHABET], 
            device=self.embed_device
        )
        self.esm_cls_id = self.model.esm_dict_cls_idx
        self.esm_eos_id = self.model.esm_dict_eos_idx
        self.esm_aa_ids = self.model.af2_to_esm[self.aa_ids + 1]

    def seq_to_logits(self, seq):
        """Convert sequence string to one-hot logits."""
        logits = torch.zeros(1, len(seq), 20, device=self.embed_device)
        for i, aa in enumerate(seq):
            if aa in AA_ALPHABET:
                idx = AA_ALPHABET.index(aa)
                logits[0, i, idx] = 10.0  # Strong bias
        return logits

    def forward_soft(self, logits, num_recycles=None):
        """Differentiable forward pass through ESMFold."""
        B, L, _ = logits.shape
        probs = F.softmax(logits, dim=-1)
        
        # ESM-2 backbone
        esm_weight = self.model.esm.embeddings.word_embeddings.weight
        probs = probs.to(esm_weight.dtype)
        soft_esm_emb = torch.matmul(probs, esm_weight[self.esm_aa_ids])
        
        cls_emb = esm_weight[self.esm_cls_id].view(1, 1, -1).expand(B, 1, -1)
        eos_emb = esm_weight[self.esm_eos_id].view(1, 1, -1).expand(B, 1, -1)
        inputs_embeds_esm = torch.cat([cls_emb, soft_esm_emb, eos_emb], dim=1)
        
        esm_outputs = self.model.esm(inputs_embeds=inputs_embeds_esm, output_hidden_states=True)
        esm_s = torch.stack(esm_outputs.hidden_states, dim=2)
        esm_s = esm_s[:, 1:-1]
        
        esm_s = esm_s.to(self.model.esm_s_combine.dtype)
        esm_s = (self.model.esm_s_combine.softmax(0).unsqueeze(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.model.esm_s_mlp(esm_s)
        
        # Trunk embeddings
        trunk_weight = self.model.embedding.weight
        soft_trunk_emb = torch.matmul(probs.to(trunk_weight.dtype), trunk_weight[self.aa_ids])
        s_s_0 += soft_trunk_emb
        
        # Trunk
        aa_input = self.aa_ids[torch.argmax(logits, dim=-1)]
        attention_mask = torch.ones((B, L), device=self.device)
        position_ids = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        s_z_0 = s_s_0.new_zeros(B, L, L, self.model.config.esmfold_config.trunk.pairwise_state_dim)
        
        structure = self.model.trunk(s_s_0, s_z_0, aa_input, position_ids, attention_mask, no_recycles=num_recycles)
        
        structure["aatype"] = aa_input
        structure["residue_index"] = position_ids
        make_atom14_masks(structure)
        
        # Prediction heads
        disto_logits = self.model.distogram_head(structure["s_z"])
        disto_logits = (disto_logits + disto_logits.transpose(1, 2)) / 2
        structure["distogram_logits"] = disto_logits
        structure["lm_logits"] = self.model.lm_head(structure["s_s"])
        
        # pLDDT
        lddt_head_output = self.model.lddt_head(structure["states"])
        lddt_head_output = lddt_head_output.reshape(
            structure["states"].shape[0], B, L, -1, self.model.lddt_bins
        )
        structure["lddt_head"] = lddt_head_output
        plddt = categorical_lddt(lddt_head_output[-1], bins=self.model.lddt_bins)
        structure["plddt"] = plddt
        
        return structure

    def extract_ca_coords(self, structure):
        """Extract CA coordinates from structure output."""
        # positions shape: (8 recycles, B, L, 14, 3) or similar
        # atom index 1 is CA
        positions = structure["positions"][-1]  # Last recycle
        ca_coords = positions[0, :, 1, :].detach().cpu().numpy()  # (L, 3)
        return ca_coords

    def design_linker(
        self, 
        pdb_path,
        chain_a_id="A",
        chain_c_id="C",
        chain_a_start_residue=None,
        chain_c_max_residue=100,
        linker_length=10,
        steps=100,
        lr=0.1,
        plddt_weight=1.0,
        tm_weight=1.0,
        out_dir="linker_design_results"
    ):
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Parse PDB
        print(f"Parsing {pdb_path}...")
        seq_a, coords_a = parse_pdb(pdb_path, chain_a_id, min_residue=chain_a_start_residue)
        seq_c, coords_c = parse_pdb(pdb_path, chain_c_id, max_residue=chain_c_max_residue)
        
        chain_a_desc = f"{chain_a_start_residue}-end" if chain_a_start_residue else "full"
        print(f"Chain {chain_a_id} ({chain_a_desc}): {len(seq_a)} residues")
        print(f"Chain {chain_c_id} (1-{chain_c_max_residue}): {len(seq_c)} residues")
        
        # Ground truth coordinates (Chain A + Chain C)
        gt_coords = np.concatenate([coords_a, coords_c], axis=0)
        gt_seq = seq_a + seq_c
        
        # Build full sequence: Chain_A + LINKER + Chain_C
        len_a = len(seq_a)
        len_c = len(seq_c)
        total_len = len_a + linker_length + len_c
        
        # Initialize logits
        # Fixed regions get strong bias, linker is random
        logits = torch.zeros(1, total_len, 20, device=self.embed_device)
        
        # Set Chain A (fixed)
        for i, aa in enumerate(seq_a):
            if aa in AA_ALPHABET:
                logits[0, i, AA_ALPHABET.index(aa)] = 10.0
        
        # Set Linker (random initialization - will be optimized)
        linker_start = len_a
        linker_end = len_a + linker_length
        logits[0, linker_start:linker_end] = torch.randn(linker_length, 20, device=self.embed_device) * 0.1
        
        # Set Chain C (fixed)
        for i, aa in enumerate(seq_c):
            if aa in AA_ALPHABET:
                logits[0, linker_end + i, AA_ALPHABET.index(aa)] = 10.0
        
        # Create mask for linker positions (only these get gradients)
        linker_mask = torch.zeros(total_len, dtype=torch.bool, device=self.embed_device)
        linker_mask[linker_start:linker_end] = True
        fixed_mask = ~linker_mask
        
        # Split logits into fixed and trainable parts
        logits_fixed = logits.clone().detach()
        logits_trainable = logits[:, linker_start:linker_end].clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([logits_trainable], lr=lr)
        
        history = []
        best_score = -float('inf')
        pbar = tqdm(range(steps))
        
        print(f"\nDesigning {linker_length}-residue linker...")
        print(f"Total sequence length: {total_len}")
        print(f"Linker positions: {linker_start+1}-{linker_end} (1-indexed)")
        
        for step in pbar:
            optimizer.zero_grad()
            
            # Reconstruct full logits
            full_logits = logits_fixed.clone()
            full_logits[:, linker_start:linker_end] = logits_trainable
            
            # Forward pass
            structure = self.forward_soft(full_logits)
            
            # Loss 1: Linker pLDDT (maximize)
            plddt = structure["plddt"]  # (B, L, 37)
            linker_plddt = plddt[0, linker_start:linker_end].mean()
            plddt_loss = -linker_plddt
            
            # Loss 2: TM-score to ground truth (maximize)
            pred_coords = self.extract_ca_coords(structure)
            
            # For TM-score, we compare only the scaffold regions (not linker)
            # Chain A coords + Chain C coords from prediction
            pred_scaffold_coords = np.concatenate([
                pred_coords[:len_a],
                pred_coords[linker_end:]
            ], axis=0)
            
            pred_seq = "".join([AA_ALPHABET[idx] for idx in torch.argmax(full_logits[0], dim=-1).cpu().numpy()])
            scaffold_seq = pred_seq[:len_a] + pred_seq[linker_end:]
            
            try:
                tm_score = compute_tm_score(pred_scaffold_coords, gt_coords, scaffold_seq, gt_seq)
            except:
                tm_score = 0.0
            
            # TM-score is not differentiable, so we use it for logging only
            # Instead, use normalized RMSD loss for scaffold regions
            pred_a_coords_t = structure["positions"][-1][0, :len_a, 1, :]  # CA of Chain A
            pred_c_coords_t = structure["positions"][-1][0, linker_end:, 1, :]  # CA of Chain C
            
            gt_a_t = torch.tensor(coords_a, device=pred_a_coords_t.device, dtype=pred_a_coords_t.dtype)
            gt_c_t = torch.tensor(coords_c, device=pred_c_coords_t.device, dtype=pred_c_coords_t.dtype)
            
            # Compute RMSD (normalized, in Angstroms) - much better scaling
            rmsd_a = torch.sqrt(((pred_a_coords_t - gt_a_t) ** 2).sum(dim=-1).mean())
            rmsd_c = torch.sqrt(((pred_c_coords_t - gt_c_t) ** 2).sum(dim=-1).mean())
            rmsd_loss = (rmsd_a + rmsd_c) / 2.0  # Average RMSD in Angstroms (~10-50 range)
            
            # Normalize RMSD to similar scale as pLDDT (0-1 range)
            # Typical RMSD: 0-50Å, so divide by 50 to get 0-1 scale
            rmsd_loss_normalized = rmsd_loss / 50.0
            
            # Combined loss: both terms now in ~0-1 range
            # plddt_loss is negative (we minimize), rmsd_loss_normalized is positive
            total_loss = plddt_weight * plddt_loss + tm_weight * rmsd_loss_normalized
            
            total_loss.backward()
            optimizer.step()
            
            # Logging
            current_linker_plddt = -plddt_loss.item()
            combined_score = current_linker_plddt + tm_score
            
            pbar.set_description(
                f"pLDDT: {current_linker_plddt:.3f} | TM: {tm_score:.3f} | RMSD: {rmsd_loss.item():.1f}Å"
            )
            
            if step % 10 == 0 or step == steps - 1:
                linker_seq = "".join([
                    AA_ALPHABET[idx] for idx in torch.argmax(logits_trainable[0], dim=-1).cpu().numpy()
                ])
                full_seq = seq_a + linker_seq + seq_c
                
                # Save PDB
                with torch.no_grad():
                    pdb_str = self.model.output_to_pdb(structure)[0]
                with open(out_path / f"step_{step:04d}.pdb", "w") as f:
                    f.write(pdb_str)
                
                history.append({
                    "step": step,
                    "linker_plddt": current_linker_plddt,
                    "tm_score": tm_score,
                    "rmsd": rmsd_loss.item(),
                    "linker_seq": linker_seq,
                    "full_seq": full_seq
                })
                
                if combined_score > best_score:
                    best_score = combined_score
                    with open(out_path / "best.pdb", "w") as f:
                        f.write(pdb_str)
        
        # Save final results
        final_linker = "".join([
            AA_ALPHABET[idx] for idx in torch.argmax(logits_trainable[0], dim=-1).cpu().numpy()
        ])
        final_seq = seq_a + final_linker + seq_c
        
        with open(out_path / "final_best.pdb", "w") as f:
            f.write(pdb_str)
        
        with open(out_path / "result.txt", "w") as f:
            f.write(f"Chain A sequence ({len_a} aa):\n{seq_a}\n\n")
            f.write(f"Designed linker ({linker_length} aa):\n{final_linker}\n\n")
            f.write(f"Chain C sequence ({len_c} aa):\n{seq_c}\n\n")
            f.write(f"Full sequence ({total_len} aa):\n{final_seq}\n\n")
            f.write(f"Final linker pLDDT: {current_linker_plddt:.4f}\n")
            f.write(f"Final TM-score: {tm_score:.4f}\n")
        
        with open(out_path / "history.json", "w") as f:
            json.dump(history, f, indent=4)
        
        # Save FASTA
        with open(out_path / "designed.fasta", "w") as f:
            f.write(f">designed_linker_plddt_{current_linker_plddt:.4f}_tm_{tm_score:.4f}\n")
            f.write(f"{final_seq}\n")
        
        print(f"\n{'='*50}")
        print(f"Linker Design Complete!")
        print(f"{'='*50}")
        print(f"Designed linker: {final_linker}")
        print(f"Linker pLDDT: {current_linker_plddt:.4f}")
        print(f"TM-score (scaffold): {tm_score:.4f}")
        print(f"Results saved to: {out_path}")
        
        return final_linker, final_seq


def main():
    parser = argparse.ArgumentParser(description="Design rigid linker between protein chains")
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB file")
    parser.add_argument("--chain_a", type=str, default="A", help="First chain ID")
    parser.add_argument("--chain_c", type=str, default="C", help="Second chain ID")
    parser.add_argument("--chain_a_start", type=int, default=None, help="Start residue for chain A (to truncate N-terminus)")
    parser.add_argument("--chain_c_max", type=int, default=100, help="Max residue for chain C")
    parser.add_argument("--linker_len", type=int, default=10, help="Linker length")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--plddt_weight", type=float, default=1.0, help="pLDDT loss weight")
    parser.add_argument("--tm_weight", type=float, default=1.0, help="TM/coord loss weight")
    parser.add_argument("--out_dir", type=str, default="linker_results", help="Output directory")
    parser.add_argument("--weights", type=str, default="/home/ubuntu/esm_weights")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing for memory efficiency")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use")
    
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    designer = LinkerDesigner(args.weights, device=device, gradient_checkpointing=args.gradient_checkpointing)
    designer.design_linker(
        pdb_path=args.pdb,
        chain_a_id=args.chain_a,
        chain_c_id=args.chain_c,
        chain_a_start_residue=args.chain_a_start,
        chain_c_max_residue=args.chain_c_max,
        linker_length=args.linker_len,
        steps=args.steps,
        lr=args.lr,
        plddt_weight=args.plddt_weight,
        tm_weight=args.tm_weight,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()

