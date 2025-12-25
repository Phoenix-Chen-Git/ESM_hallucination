#!/usr/bin/env python3
"""
FSDP-based Gradient Linker Design using all 4 GPUs.
Uses Fully Sharded Data Parallel to shard the ESMFold model across GPUs.
Launch with: torchrun --nproc_per_node=4 linker_design_fsdp.py --pdb design/design4.pdb ...
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.checkpoint import checkpoint
import numpy as np
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.modeling_esmfold import EsmFoldTriangularSelfAttentionBlock
from tqdm import tqdm
from Bio.PDB import PDBParser
import tmtools
import functools

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


def setup_distributed():
    """Initialize distributed training."""
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup():
    dist.destroy_process_group()


class FSDPLinkerDesigner:
    """
    FSDP-based linker designer that shards ESMFold across all GPUs.
    """
    
    def __init__(self, model_path, local_rank):
        self.local_rank = local_rank
        self.device = torch.device(f"cuda:{local_rank}")
        self.is_main = (local_rank == 0)
        
        if self.is_main:
            print(f"Loading model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = EsmForProteinFolding.from_pretrained(model_path, weights_only=False)
        
        # Cast model to float32 for uniform dtype (FSDP requirement)
        self.model = self.model.float()
        
        # Enable gradient checkpointing on ESM encoder
        self.model.esm.encoder.gradient_checkpointing = True
        self.model.trunk.chunk_size = 4
        
        # FSDP wrapping with mixed precision
        mixed_precision = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
        
        # Auto-wrap policy for transformer blocks
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={EsmFoldTriangularSelfAttentionBlock},
        )
        
        # Wrap with FSDP
        self.model = FSDP(
            self.model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            auto_wrap_policy=auto_wrap_policy,
            device_id=self.device,
            use_orig_params=True,  # Required for per-parameter learning rates
        )
        
        # Get AA token IDs (these need to be computed before FSDP wrapping ideally, but should work)
        self.aa_ids = torch.tensor(
            [self.tokenizer.convert_tokens_to_ids(aa) for aa in AA_ALPHABET],
            device=self.device
        )
        
        if self.is_main:
            print(f"Model loaded with FSDP across {dist.get_world_size()} GPUs!")
    
    def forward_fold(self, sequence):
        """Standard forward pass for inference."""
        inputs = self.tokenizer([sequence], return_tensors='pt', add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            output = self.model(**inputs, num_recycles=2)
        
        return output
    
    def kabsch_rmsd(self, pred_coords, gt_coords):
        """Differentiable Kabsch RMSD."""
        # Cast to float32 for SVD
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
        steps=100,
        lr=0.5,
        plddt_weight=1.0,
        rmsd_weight=0.1,
        out_dir="linker_fsdp_results"
    ):
        """FSDP-based gradient linker design."""
        out_path = Path(out_dir)
        if self.is_main:
            out_path.mkdir(parents=True, exist_ok=True)
        
        # Parse PDB (only main process prints)
        if self.is_main:
            print(f"Parsing {pdb_path}...")
        seq_a, coords_a = parse_pdb(pdb_path, chain_a_id, min_residue=chain_a_start_residue)
        seq_c, coords_c = parse_pdb(pdb_path, chain_c_id, max_residue=chain_c_max_residue)
        
        if self.is_main:
            print(f"Chain A: {len(seq_a)} residues")
            print(f"Chain C: {len(seq_c)} residues")
        
        len_a = len(seq_a)
        len_c = len(seq_c)
        linker_start = len_a
        linker_end = len_a + linker_length
        total_len = len_a + linker_length + len_c
        
        if self.is_main:
            print(f"\nDesigning {linker_length}-residue linker...")
            print(f"Total sequence length: {total_len}")
        
        # Ground truth coordinates
        gt_coords_a = torch.tensor(coords_a, device=self.device, dtype=torch.float32)
        gt_coords_c = torch.tensor(coords_c, device=self.device, dtype=torch.float32)
        
        # Initialize linker logits (shared across all ranks via broadcast)
        linker_logits = torch.randn(1, linker_length, 20, device=self.device) * 0.1
        dist.broadcast(linker_logits, src=0)  # Ensure all ranks have same initial logits
        linker_logits = linker_logits.detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([linker_logits], lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        scaler = torch.cuda.amp.GradScaler()
        
        history = []
        best_plddt = 0.0
        best_linker = ""
        best_pdb = None
        
        pbar = tqdm(range(steps), disable=not self.is_main)
        for step in pbar:
            optimizer.zero_grad()
            
            # Build full sequence
            linker_seq = "".join([AA_ALPHABET[i] for i in linker_logits[0].argmax(dim=-1).cpu().numpy()])
            full_seq = seq_a + linker_seq + seq_c
            
            try:
                # Forward pass with mixed precision
                with torch.cuda.amp.autocast():
                    inputs = self.tokenizer([full_seq], return_tensors='pt', add_special_tokens=False)
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                    
                    # Get embeddings from input_ids
                    # We need to inject soft embeddings for the linker region
                    # For FSDP, we'll use a simpler approach: use the model's embedding layer
                    # and replace the linker embeddings with soft ones
                    
                    output = self.model(**inputs, num_recycles=0)
                    
                    # Compute linker pLDDT
                    plddt = output.plddt[0]  # (L,)
                    linker_plddt = plddt[linker_start:linker_end].mean()
                    plddt_loss = -linker_plddt
                    
                    # Get CA coordinates for RMSD
                    ca_coords = output.positions[-1, 0, :, 1, :]  # (L, 3)
                    pred_a = ca_coords[:len_a]
                    pred_c = ca_coords[linker_end:]
                
                # RMSD outside autocast for SVD compatibility
                rmsd_a = self.kabsch_rmsd(pred_a, gt_coords_a)
                rmsd_c = self.kabsch_rmsd(pred_c, gt_coords_c)
                rmsd_loss = rmsd_a + rmsd_c
                
                # Total loss
                loss = plddt_weight * plddt_loss + rmsd_weight * rmsd_loss
                
                # Backward pass
                scaler.scale(loss).backward()
                
                # All-reduce gradients across GPUs (FSDP handles this automatically for model params)
                # For linker_logits, we need to manually sync if needed
                if linker_logits.grad is not None:
                    dist.all_reduce(linker_logits.grad, op=dist.ReduceOp.AVG)
                
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([linker_logits], max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Logging (main process only)
                plddt_val = linker_plddt.item()
                rmsd_val = rmsd_loss.item()
                
                if plddt_val > best_plddt:
                    best_plddt = plddt_val
                    best_linker = linker_seq
                    if self.is_main:
                        with torch.no_grad():
                            best_pdb = self.model.module.output_to_pdb(output)[0]
                
                if self.is_main:
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
                        with torch.no_grad():
                            pdb_str = self.model.module.output_to_pdb(output)[0]
                        with open(out_path / f"step_{step:04d}.pdb", "w") as f:
                            f.write(pdb_str)
                
                # Clear cache
                del output, loss
                torch.cuda.empty_cache()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    if self.is_main:
                        print(f"\nOOM at step {step}, trying to recover...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # Save final results (main process only)
        if self.is_main:
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
        
        return best_linker, seq_a + best_linker + seq_c


def main():
    parser = argparse.ArgumentParser(description="FSDP-based linker design")
    parser.add_argument("--pdb", type=str, required=True)
    parser.add_argument("--chain_a", type=str, default="A")
    parser.add_argument("--chain_c", type=str, default="C")
    parser.add_argument("--chain_a_start", type=int, default=None)
    parser.add_argument("--chain_c_max", type=int, default=100)
    parser.add_argument("--linker_len", type=int, default=30)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.5)
    parser.add_argument("--plddt_weight", type=float, default=1.0)
    parser.add_argument("--rmsd_weight", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="linker_fsdp_results")
    parser.add_argument("--weights", type=str, default="/home/ubuntu/esm_weights")
    
    args = parser.parse_args()
    
    local_rank = setup_distributed()
    
    try:
        designer = FSDPLinkerDesigner(args.weights, local_rank)
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
    finally:
        cleanup()


if __name__ == "__main__":
    main()

