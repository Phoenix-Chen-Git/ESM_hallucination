#!/usr/bin/env python3
"""
Multi-GPU Gradient-Based Linker Design
Uses DeepSpeed ZeRO or manual gradient accumulation across GPUs for backpropagation.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
import numpy as np
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, EsmForProteinFolding
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
                break
        break
    
    return "".join(sequence), np.array(ca_coords)


def compute_tm_score(coords1, coords2, seq1, seq2):
    """Compute TM-score between two structures."""
    min_len = min(len(coords1), len(coords2))
    coords1 = coords1[:min_len]
    coords2 = coords2[:min_len]
    result = tmtools.tm_align(coords1, coords2, seq1[:min_len], seq2[:min_len])
    return result.tm_norm_chain1


class GradientLinkerDesigner:
    """
    Gradient-based linker designer using activation checkpointing and 
    gradient accumulation across multiple GPUs.
    """
    
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        print(f"Loading model on {device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = EsmForProteinFolding.from_pretrained(model_path, weights_only=False)
        self.model.to(device)
        
        # Enable gradient checkpointing for memory efficiency
        self.model.esm.encoder.gradient_checkpointing = True
        self.model.trunk.chunk_size = 4
        self.model.trunk.no_recycles = 0
        
        # Monkey-patch ESMFold trunk to use gradient checkpointing on each block
        # This is much more memory efficient than checkpointing the whole trunk
        def block_forward_with_checkpoint(self_block, *args, **kwargs):
            return checkpoint(self_block.__class__.forward, self_block, *args, **kwargs, use_reentrant=False)
        
        for block in self.model.trunk.blocks:
            block.forward = block_forward_with_checkpoint.__get__(block, block.__class__)
        
        print("Model loaded with aggressive per-block gradient checkpointing!")
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
            # Skip mask token handling when using inputs_embeds
            return embeddings
        
        self.model.esm.embeddings.forward = patched_forward.__get__(
            self.model.esm.embeddings, self.model.esm.embeddings.__class__
        )
        
        # Get AA token IDs for soft embedding
        self.aa_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(aa) for aa in AA_ALPHABET],
            device=device
        )
        
        # ESM AA indices using model's mapping
        self.esm_cls_id = self.model.esm_dict_cls_idx
        self.esm_eos_id = self.model.esm_dict_eos_idx
        self.esm_aa_ids = self.model.af2_to_esm[self.aa_ids + 1]
        
        print("Model loaded with gradient checkpointing enabled!")
    
    def seq_to_logits(self, seq):
        """Convert sequence string to one-hot logits."""
        logits = torch.zeros(1, len(seq), 20, device=self.device)
        for i, aa in enumerate(seq):
            if aa in AA_ALPHABET:
                idx = AA_ALPHABET.index(aa)
                logits[0, i, idx] = 10.0
        return logits
    
    def forward_soft(self, logits, num_recycles=1):
        """Differentiable forward pass through ESMFold."""
        B, L, _ = logits.shape
        probs = F.softmax(logits, dim=-1)
        
        # ESM-2 backbone with soft embeddings
        esm_weight = self.model.esm.embeddings.word_embeddings.weight
        probs = probs.to(esm_weight.dtype)
        soft_esm_emb = torch.matmul(probs, esm_weight[self.esm_aa_ids])
        
        # Add positional embeddings
        position_ids = torch.arange(L, device=self.device).unsqueeze(0)
        
        # BOS/EOS tokens using model's indices
        bos_emb = esm_weight[self.esm_cls_id].unsqueeze(0).unsqueeze(0)
        eos_emb = esm_weight[self.esm_eos_id].unsqueeze(0).unsqueeze(0)
        inputs_embeds_esm = torch.cat([bos_emb, soft_esm_emb, eos_emb], dim=1)
        
        # Run ESM encoder with patched embeddings (no input_ids needed)
        esm_outputs = self.model.esm(
            inputs_embeds=inputs_embeds_esm, 
            output_hidden_states=True
        )
        esm_s = torch.stack(esm_outputs.hidden_states, dim=2)
        esm_s = esm_s[:, 1:-1]  # Remove BOS/EOS
        
        esm_s = esm_s.to(self.model.esm_s_combine.dtype)
        esm_s = (self.model.esm_s_combine.softmax(0).unsqueeze(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.model.esm_s_mlp(esm_s)
        
        # Trunk embeddings
        trunk_weight = self.model.embedding.weight
        soft_trunk_emb = torch.matmul(probs.to(trunk_weight.dtype), trunk_weight[self.aa_ids])
        s_s_0 += soft_trunk_emb
        
        # Trunk forward
        aa_input = self.aa_ids[torch.argmax(logits, dim=-1)]
        attention_mask = torch.ones((B, L), device=self.device)
        position_ids = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
        s_z_0 = s_s_0.new_zeros(B, L, L, self.model.config.esmfold_config.trunk.pairwise_state_dim)
        
        # Now calling trunk normally - blocks are internally checkpointed
        structure = self.model.trunk(s_s_0, s_z_0, aa_input, position_ids, attention_mask, no_recycles=num_recycles)
        
        structure["aatype"] = aa_input
        structure["residue_index"] = position_ids
        
        # Build atom14 masks
        from transformers.models.esm.openfold_utils import make_atom14_masks
        make_atom14_masks(structure)
        
        # Get pLDDT from lddt_head
        lddt_head_output = self.model.lddt_head(structure["states"])
        lddt_head_output = lddt_head_output.reshape(
            structure["states"].shape[0], B, L, -1, self.model.lddt_bins
        )
        
        # Compute pLDDT from categorical output
        def categorical_lddt(logits, bins=50):
            probs = F.softmax(logits, dim=-1)
            centers = torch.linspace(1.0 / bins / 2, 1.0 - 1.0 / bins / 2, bins, device=logits.device)
            return (probs * centers).sum(dim=-1)
        
        plddt = categorical_lddt(lddt_head_output[-1], bins=self.model.lddt_bins)
        
        # Get positions
        positions = structure["positions"]
        
        # Add plddt to structure for output_to_pdb
        structure["plddt"] = plddt
        
        return {
            "plddt": plddt,
            "positions": positions,
            "structure": structure
        }
    
    def kabsch_rmsd(self, pred_coords, gt_coords):
        """Differentiable Kabsch RMSD."""
        # Cast to float32 for SVD (doesn't work with half precision)
        pred_coords = pred_coords.float()
        gt_coords = gt_coords.float()
        
        # Center
        pred_center = pred_coords.mean(dim=0, keepdim=True)
        gt_center = gt_coords.mean(dim=0, keepdim=True)
        pred_centered = pred_coords - pred_center
        gt_centered = gt_coords - gt_center
        
        # Covariance
        H = pred_centered.T @ gt_centered
        U, S, Vt = torch.linalg.svd(H)
        
        # Handle reflection
        d = torch.det(Vt.T @ U.T)
        sign_matrix = torch.diag(torch.tensor([1., 1., d.sign()], device=pred_coords.device))
        
        # Rotation
        R = Vt.T @ sign_matrix @ U.T
        pred_aligned = pred_centered @ R
        
        # RMSD
        diff = pred_aligned - gt_centered
        rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean())
        return rmsd
    
    def design_linker(
        self,
        pdb_path,
        chain_a_id="A",
        chain_c_id="C",
        chain_a_start_residue=None,
        chain_c_max_residue=100,
        linker_length=30,
        steps=200,
        lr=0.5,
        plddt_weight=1.0,
        rmsd_weight=0.1,
        out_dir="linker_gradient_results"
    ):
        """Gradient-based linker design."""
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Parse PDB
        print(f"Parsing {pdb_path}...")
        seq_a, coords_a = parse_pdb(pdb_path, chain_a_id, min_residue=chain_a_start_residue)
        seq_c, coords_c = parse_pdb(pdb_path, chain_c_id, max_residue=chain_c_max_residue)
        
        print(f"Chain A: {len(seq_a)} residues")
        print(f"Chain C: {len(seq_c)} residues")
        
        len_a = len(seq_a)
        len_c = len(seq_c)
        linker_start = len_a
        linker_end = len_a + linker_length
        total_len = len_a + linker_length + len_c
        
        print(f"\nDesigning {linker_length}-residue linker...")
        print(f"Total sequence length: {total_len}")
        
        # Ground truth coordinates
        gt_coords_a = torch.tensor(coords_a, device=self.device, dtype=torch.float32)
        gt_coords_c = torch.tensor(coords_c, device=self.device, dtype=torch.float32)
        
        # Initialize logits
        logits = torch.zeros(1, total_len, 20, device=self.device, requires_grad=False)
        
        # Fixed regions (Chain A and C)
        for i, aa in enumerate(seq_a):
            if aa in AA_ALPHABET:
                logits[0, i, AA_ALPHABET.index(aa)] = 10.0
        for i, aa in enumerate(seq_c):
            if aa in AA_ALPHABET:
                logits[0, linker_end + i, AA_ALPHABET.index(aa)] = 10.0
        
        # Linker region (learnable)
        linker_logits = torch.randn(1, linker_length, 20, device=self.device) * 0.1
        linker_logits.requires_grad_(True)
        
        optimizer = torch.optim.Adam([linker_logits], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        
        # Mixed precision scaler for memory efficiency
        scaler = torch.cuda.amp.GradScaler()
        
        history = []
        best_plddt = 0
        best_linker = ""
        best_pdb = None
        
        pbar = tqdm(range(steps))
        for step in pbar:
            optimizer.zero_grad()
            
            # Combine fixed and learnable logits
            full_logits = logits.clone()
            full_logits[0, linker_start:linker_end] = linker_logits[0]
            
            try:
                # Forward pass with mixed precision (0 recycles to save memory)
                with torch.cuda.amp.autocast():
                    output = self.forward_soft(full_logits, num_recycles=0)
                    
                    # Linker pLDDT loss (maximize)
                    linker_plddt = output["plddt"][0, linker_start:linker_end].mean()
                    plddt_loss = -linker_plddt
                    
                    # Structure alignment loss (minimize RMSD to ground truth)
                    pred_coords = output["positions"][-1, 0, :, 1, :]  # CA coords
                    pred_a = pred_coords[:len_a]
                    pred_c = pred_coords[linker_end:]
                
                # Move RMSD out of autocast to avoid SVD Half error
                rmsd_a = self.kabsch_rmsd(pred_a, gt_coords_a)
                rmsd_c = self.kabsch_rmsd(pred_c, gt_coords_c)
                rmsd_loss = rmsd_a + rmsd_c
                
                # Total loss
                loss = plddt_weight * plddt_loss + rmsd_weight * rmsd_loss
                
                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([linker_logits], max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Get current linker sequence
                linker_seq = "".join([AA_ALPHABET[i] for i in linker_logits[0].argmax(dim=-1).cpu().numpy()])
                
                plddt_val = linker_plddt.item()
                rmsd_val = rmsd_loss.item()
                
                if plddt_val > best_plddt:
                    best_plddt = plddt_val
                    best_linker = linker_seq
                    # Generate PDB
                    with torch.no_grad():
                        full_seq = seq_a + linker_seq + seq_c
                        inputs = self.tokenizer([full_seq], return_tensors='pt', add_special_tokens=False)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        out = self.model(**inputs, num_recycles=2)
                        best_pdb = self.model.output_to_pdb(out)[0]
                
                pbar.set_description(f"pLDDT: {plddt_val:.3f} | RMSD: {rmsd_val:.2f} | Best: {best_plddt:.3f}")
                
                if step % 10 == 0:
                    history.append({
                        "step": step,
                        "linker_plddt": plddt_val,
                        "rmsd": rmsd_val,
                        "linker_seq": linker_seq,
                        "lr": scheduler.get_last_lr()[0]
                    })
                
                # Save PDB checkpoint
                if step % 10 == 0 or step == steps - 1:
                    with torch.no_grad():
                        # Generate PDB using discrete sequence
                        full_seq = seq_a + linker_seq + seq_c
                        inputs = self.tokenizer([full_seq], return_tensors='pt', add_special_tokens=False)
                        inputs = {k: v.to(self.device) for k, v in inputs.items()}
                        pdb_out = self.model(**inputs, num_recycles=1)
                        pdb_str = self.model.output_to_pdb(pdb_out)[0]
                    with open(out_path / f"step_{step:04d}.pdb", "w") as f:
                        f.write(pdb_str)
                
                # Clear cache
                del output, loss, plddt_loss, rmsd_loss
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nOOM at step {step}, trying to recover...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Save results
        final_seq = seq_a + best_linker + seq_c
        
        if best_pdb:
            with open(out_path / "best.pdb", "w") as f:
                f.write(best_pdb)
        
        with open(out_path / "result.txt", "w") as f:
            f.write(f"Designed linker: {best_linker}\n")
            f.write(f"Best pLDDT: {best_plddt:.4f}\n")
            f.write(f"Full sequence:\n{final_seq}\n")
        
        with open(out_path / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        print(f"\n{'='*50}")
        print(f"Design Complete!")
        print(f"Best linker: {best_linker}")
        print(f"Best pLDDT: {best_plddt:.4f}")
        print(f"Results saved to: {out_path}")
        
        return best_linker, final_seq


def main():
    parser = argparse.ArgumentParser(description="Gradient-based linker design")
    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--chain_a", type=str, default="A")
    parser.add_argument("--chain_c", type=str, default="C")
    parser.add_argument("--chain_a_start", type=int, default=None)
    parser.add_argument("--chain_c_max", type=int, default=100)
    parser.add_argument("--linker_len", type=int, default=30)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--plddt_weight", type=float, default=1.0)
    parser.add_argument("--rmsd_weight", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="linker_gradient_results")
    parser.add_argument("--weights", type=str, default="/home/ubuntu/esm_weights")
    parser.add_argument("--gpu", type=int, default=0)
    
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}"
    designer = GradientLinkerDesigner(args.weights, device=device)
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
        rmsd_weight=args.rmsd_weight,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()

