#!/usr/bin/env python3
"""
Linker Design Script - Flexible AID Version
Design a rigid linker between AID and Pcra using ESMFold.
Optimizes for: (1) High pLDDT of linker (rigidity)
               (2) No steric clashes between AID and Pcra
               (3) Active site accessibility (catalytic residues exposed)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np
import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer, EsmForProteinFolding
from transformers.models.esm.openfold_utils import make_atom14_masks
from transformers.models.esm.modeling_esmfold import categorical_lddt
from tqdm import tqdm
from Bio.PDB import PDBParser

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


def compute_clash_penalty(coords_a, coords_c, threshold=4.0):
    """
    Compute clash penalty between AID (Chain A) and Pcra (Chain C).
    Penalizes Cα-Cα distances below threshold.
    
    Args:
        coords_a: (L_a, 3) tensor of AID Cα coordinates
        coords_c: (L_c, 3) tensor of Pcra Cα coordinates
        threshold: distance threshold in Angstroms (default 4.0)
    
    Returns:
        Scalar penalty (higher = more clashes)
    """
    # Compute pairwise distances
    dists = torch.cdist(coords_a, coords_c)  # (L_a, L_c)
    # Soft penalty: ReLU(threshold - dist), so violations contribute positively
    violations = F.relu(threshold - dists)
    # Sum all violations (could also use mean)
    return violations.sum()


def compute_active_site_exposure(coords_a, coords_c, active_residues):
    """
    Compute exposure score for active site residues.
    Higher score = active site is farther from Pcra = more accessible.
    
    Args:
        coords_a: (L_a, 3) tensor of AID Cα coordinates
        coords_c: (L_c, 3) tensor of Pcra Cα coordinates
        active_residues: list of 0-indexed residue positions in AID
    
    Returns:
        Mean minimum distance from active site to Pcra (higher = more exposed)
    """
    # Get active site coordinates
    active_coords = coords_a[active_residues]  # (n_active, 3)
    # Distance from each active residue to all Pcra residues
    dists = torch.cdist(active_coords, coords_c)  # (n_active, L_c)
    # Minimum distance to any Pcra residue for each active site residue
    min_dists = dists.min(dim=1).values  # (n_active,)
    # Mean of minimum distances
    return min_dists.mean()


class LinkerDesigner:
    def __init__(self, model_path, device=None, multi_gpu=False, gradient_checkpointing=False):
        self.multi_gpu = multi_gpu
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.gradient_checkpointing = gradient_checkpointing
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = EsmForProteinFolding.from_pretrained(model_path, weights_only=False)
        self.model.to(self.device)
        
        if gradient_checkpointing:
            print("Enabling aggressive gradient checkpointing...")
            
            # 1. Enable gradient checkpointing for ESM-2 encoder
            self.model.esm.encoder.gradient_checkpointing = True
            print("  - ESM encoder checkpointing enabled")
            
            # 2. Set small chunk size for trunk
            if hasattr(self.model, 'trunk') and hasattr(self.model.trunk, 'chunk_size'):
                self.model.trunk.chunk_size = 4
                print("  - Trunk chunk_size set to 4")
            
            # 3. KEY: Monkey-patch EVERY trunk block with per-block checkpointing
            if hasattr(self.model, 'trunk') and hasattr(self.model.trunk, 'blocks'):
                def make_checkpointed_forward(original_forward):
                    def checkpointed_forward(self_block, *args, **kwargs):
                        return checkpoint(
                            original_forward,
                            self_block,
                            *args,
                            **kwargs,
                            use_reentrant=False
                        )
                    return checkpointed_forward
                
                for i, block in enumerate(self.model.trunk.blocks):
                    original_forward = block.__class__.forward
                    block.forward = make_checkpointed_forward(original_forward).__get__(block, block.__class__)
                print(f"  - Per-block checkpointing enabled for {len(self.model.trunk.blocks)} trunk blocks")
        else:
            # Set trunk chunk_size to reduce memory for structure module
            if hasattr(self.model, 'trunk') and hasattr(self.model.trunk, 'chunk_size'):
                self.model.trunk.chunk_size = 32
                print(f"Set trunk chunk_size to 32")
        
        # Reduce number of recycles for memory efficiency
        self.num_recycles = 0 if gradient_checkpointing else 2
        print(f"  - num_recycles set to {self.num_recycles}")
        
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

    def _esm_forward(self, inputs_embeds_esm):
        """Helper for checkpointed ESM forward."""
        esm_outputs = self.model.esm(inputs_embeds=inputs_embeds_esm, output_hidden_states=True)
        esm_s = torch.stack(esm_outputs.hidden_states, dim=2)
        esm_s = esm_s[:, 1:-1]
        esm_s = esm_s.to(self.model.esm_s_combine.dtype)
        esm_s = (self.model.esm_s_combine.softmax(0).unsqueeze(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.model.esm_s_mlp(esm_s)
        return s_s_0

    def forward_soft(self, logits, num_recycles=None):
        """Differentiable forward pass through ESMFold with optional gradient checkpointing and fp16."""
        B, L, _ = logits.shape
        
        # Use automatic mixed precision for memory efficiency
        with torch.cuda.amp.autocast(enabled=self.gradient_checkpointing):
            probs = F.softmax(logits, dim=-1)
            
            # ESM-2 backbone
            esm_weight = self.model.esm.embeddings.word_embeddings.weight
            probs = probs.to(esm_weight.dtype)
            soft_esm_emb = torch.matmul(probs, esm_weight[self.esm_aa_ids])
            
            cls_emb = esm_weight[self.esm_cls_id].view(1, 1, -1).expand(B, 1, -1)
            eos_emb = esm_weight[self.esm_eos_id].view(1, 1, -1).expand(B, 1, -1)
            inputs_embeds_esm = torch.cat([cls_emb, soft_esm_emb, eos_emb], dim=1)
            
            # ESM forward (with optional checkpointing)
            if self.gradient_checkpointing:
                s_s_0 = checkpoint(self._esm_forward, inputs_embeds_esm, use_reentrant=False)
            else:
                s_s_0 = self._esm_forward(inputs_embeds_esm)
            
            # Trunk embeddings
            trunk_weight = self.model.embedding.weight
            soft_trunk_emb = torch.matmul(probs.to(trunk_weight.dtype), trunk_weight[self.aa_ids])
            s_s_0 = s_s_0 + soft_trunk_emb
            
            # Trunk
            aa_input = self.aa_ids[torch.argmax(logits, dim=-1)]
            attention_mask = torch.ones((B, L), device=self.device)
            position_ids = torch.arange(L, device=self.device).unsqueeze(0).expand(B, L)
            s_z_0 = s_s_0.new_zeros(B, L, L, self.model.config.esmfold_config.trunk.pairwise_state_dim)
            
            # Trunk forward
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
        clash_weight=1.0,
        exposure_weight=1.0,
        active_residues=None,
        clash_threshold=4.0,
        out_dir="linker_design_results"
    ):
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Parse active residues (convert 1-indexed to 0-indexed)
        if active_residues is None:
            active_residues = [55, 57, 86, 89]  # Default: H56, E58, C87, C90 (0-indexed)
        else:
            active_residues = [r - 1 for r in active_residues]  # Convert to 0-indexed
        
        # Parse PDB
        print(f"Parsing {pdb_path}...")
        seq_a, coords_a = parse_pdb(pdb_path, chain_a_id, min_residue=chain_a_start_residue)
        seq_c, coords_c = parse_pdb(pdb_path, chain_c_id, max_residue=chain_c_max_residue)
        
        chain_a_desc = f"{chain_a_start_residue}-end" if chain_a_start_residue else "full"
        print(f"Chain {chain_a_id} ({chain_a_desc}): {len(seq_a)} residues")
        print(f"Chain {chain_c_id} (1-{chain_c_max_residue}): {len(seq_c)} residues")
        print(f"Active site residues (1-indexed): {[r+1 for r in active_residues]}")
        
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
            
            # Forward pass with specified recycles
            structure = self.forward_soft(full_logits, num_recycles=self.num_recycles)
            
            # Loss 1: Linker pLDDT (maximize) - for rigidity
            plddt = structure["plddt"]  # (B, L)
            linker_plddt = plddt[0, linker_start:linker_end].mean()
            plddt_loss = -linker_plddt  # Negative because we minimize loss
            
            # Extract predicted coordinates for AID and Pcra
            pred_a_coords_t = structure["positions"][-1][0, :len_a, 1, :]  # CA of AID (L_a, 3)
            pred_c_coords_t = structure["positions"][-1][0, linker_end:, 1, :]  # CA of Pcra (L_c, 3)
            
            # Loss 2: Clash penalty (minimize) - no steric clashes
            clash_penalty = compute_clash_penalty(pred_a_coords_t, pred_c_coords_t, threshold=clash_threshold)
            # Normalize: typical clashes ~0-1000, divide by 100 to get ~0-10 scale
            clash_loss = clash_penalty / 100.0
            
            # Loss 3: Active site exposure (maximize) - catalytic site accessible
            exposure_score = compute_active_site_exposure(pred_a_coords_t, pred_c_coords_t, active_residues)
            # Normalize: typical exposure ~5-50Å, divide by 50 to get ~0-1 scale
            exposure_loss = -exposure_score / 50.0  # Negative because we maximize exposure
            
            # Combined loss
            total_loss = (
                plddt_weight * plddt_loss +
                clash_weight * clash_loss +
                exposure_weight * exposure_loss
            )
            
            total_loss.backward()
            optimizer.step()
            
            # Logging
            current_linker_plddt = -plddt_loss.item()
            current_clash = clash_penalty.item()
            current_exposure = exposure_score.item()
            
            # Score: higher pLDDT + higher exposure + lower clashes is better
            combined_score = current_linker_plddt + current_exposure / 10.0 - current_clash / 100.0
            
            pbar.set_description(
                f"pLDDT: {current_linker_plddt:.3f} | Clash: {current_clash:.1f} | Exposure: {current_exposure:.1f}Å"
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
                    "clash_penalty": current_clash,
                    "exposure_score": current_exposure,
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
            f.write(f"AID (Chain A) sequence ({len_a} aa):\n{seq_a}\n\n")
            f.write(f"Designed linker ({linker_length} aa):\n{final_linker}\n\n")
            f.write(f"Pcra (Chain C) sequence ({len_c} aa):\n{seq_c}\n\n")
            f.write(f"Full sequence ({total_len} aa):\n{final_seq}\n\n")
            f.write(f"Active site residues (1-indexed): {[r+1 for r in active_residues]}\n\n")
            f.write(f"Final linker pLDDT: {current_linker_plddt:.4f}\n")
            f.write(f"Final clash penalty: {current_clash:.4f}\n")
            f.write(f"Final active site exposure: {current_exposure:.4f} Å\n")
        
        with open(out_path / "history.json", "w") as f:
            json.dump(history, f, indent=4)
        
        # Save FASTA
        with open(out_path / "designed.fasta", "w") as f:
            f.write(f">AID_linker_Pcra_plddt_{current_linker_plddt:.2f}_clash_{current_clash:.0f}_exp_{current_exposure:.1f}\n")
            f.write(f"{final_seq}\n")
        
        print(f"\n{'='*50}")
        print(f"Flexible AID Linker Design Complete!")
        print(f"{'='*50}")
        print(f"Designed linker: {final_linker}")
        print(f"Linker pLDDT (rigidity): {current_linker_plddt:.4f}")
        print(f"Clash penalty (lower=better): {current_clash:.4f}")
        print(f"Active site exposure (higher=better): {current_exposure:.4f} Å")
        print(f"Results saved to: {out_path}")
        
        return final_linker, final_seq


def main():
    parser = argparse.ArgumentParser(description="Design flexible AID linker with clash avoidance and active site exposure")
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB file")
    parser.add_argument("--chain_a", type=str, default="A", help="AID chain ID")
    parser.add_argument("--chain_c", type=str, default="C", help="Pcra chain ID")
    parser.add_argument("--chain_a_start", type=int, default=None, help="Start residue for AID (to truncate N-terminus)")
    parser.add_argument("--chain_c_max", type=int, default=100, help="Max residue for Pcra")
    parser.add_argument("--linker_len", type=int, default=30, help="Linker length (default 30)")
    parser.add_argument("--steps", type=int, default=100, help="Optimization steps")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--plddt_weight", type=float, default=1.0, help="pLDDT (rigidity) loss weight")
    parser.add_argument("--clash_weight", type=float, default=1.0, help="Clash penalty weight")
    parser.add_argument("--exposure_weight", type=float, default=1.0, help="Active site exposure weight")
    parser.add_argument("--active_residues", type=str, default="56,58,87,90", 
                        help="Comma-separated catalytic residues (1-indexed)")
    parser.add_argument("--clash_threshold", type=float, default=4.0, help="Clash distance threshold (Å)")
    parser.add_argument("--out_dir", type=str, default="linker_results", help="Output directory")
    parser.add_argument("--weights", type=str, default="/home/ubuntu/esm_weights")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Enable gradient checkpointing")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    
    args = parser.parse_args()
    
    # Parse active residues
    active_residues = [int(r.strip()) for r in args.active_residues.split(",")]
    
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
        clash_weight=args.clash_weight,
        exposure_weight=args.exposure_weight,
        active_residues=active_residues,
        clash_threshold=args.clash_threshold,
        out_dir=args.out_dir
    )


if __name__ == "__main__":
    main()

