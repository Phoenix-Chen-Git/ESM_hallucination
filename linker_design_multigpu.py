#!/usr/bin/env python3
"""
Multi-GPU Linker Design Script
Design a rigid linker between two protein chains using ESMFold.
Uses inference-only optimization (MCMC) to handle longer sequences across multiple GPUs.
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
from tqdm import tqdm
from Bio.PDB import PDBParser
import tmtools
import random

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
    """Compute TM-score between two structures using tmtools."""
    min_len = min(len(coords1), len(coords2))
    coords1 = coords1[:min_len]
    coords2 = coords2[:min_len]
    
    result = tmtools.tm_align(coords1, coords2, seq1[:min_len], seq2[:min_len])
    return result.tm_norm_chain1


class MultiGPULinkerDesigner:
    """
    Multi-GPU linker designer using MCMC optimization.
    Distributes inference across multiple GPUs for longer sequences.
    """
    
    def __init__(self, model_path, num_gpus=4):
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        print(f"Initializing with {self.num_gpus} GPUs...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load model on each GPU
        self.models = []
        for i in range(self.num_gpus):
            print(f"Loading model on GPU {i}...")
            model = EsmForProteinFolding.from_pretrained(model_path, weights_only=False)
            model.to(f'cuda:{i}')
            model.eval()
            model.trunk.chunk_size = 32
            self.models.append(model)
        
        print(f"All {self.num_gpus} models loaded!")
    
    @torch.no_grad()
    def predict_structure(self, sequence, gpu_id=0):
        """Run structure prediction on specified GPU."""
        model = self.models[gpu_id]
        device = f'cuda:{gpu_id}'
        
        inputs = self.tokenizer([sequence], return_tensors='pt', add_special_tokens=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        output = model(**inputs, num_recycles=2)
        
        # Extract results
        plddt = output.plddt[0].cpu().numpy()
        positions = output.positions[-1][0, :, 1, :].cpu().numpy()  # CA coords
        pdb_str = model.output_to_pdb(output)[0]
        
        return {
            'plddt': plddt,
            'ca_coords': positions,
            'pdb_str': pdb_str,
            'mean_plddt': float(plddt.mean())
        }
    
    def mutate_linker(self, linker_seq, num_mutations=1):
        """Randomly mutate the linker sequence."""
        linker_list = list(linker_seq)
        for _ in range(num_mutations):
            pos = random.randint(0, len(linker_list) - 1)
            linker_list[pos] = random.choice(AA_ALPHABET)
        return "".join(linker_list)
    
    def design_linker_mcmc(
        self,
        pdb_path,
        chain_a_id="A",
        chain_c_id="C",
        chain_a_start_residue=None,
        chain_c_max_residue=100,
        linker_length=30,
        steps=200,
        temperature=1.0,
        temp_decay=0.995,
        plddt_weight=1.0,
        tm_weight=1.0,
        out_dir="linker_multigpu_results",
        base_mutations=3
    ):
        """
        MCMC-based linker design using multiple GPUs in parallel.
        Each GPU evaluates different candidate mutations.
        """
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        
        # Parse PDB
        print(f"Parsing {pdb_path}...")
        seq_a, coords_a = parse_pdb(pdb_path, chain_a_id, min_residue=chain_a_start_residue)
        seq_c, coords_c = parse_pdb(pdb_path, chain_c_id, max_residue=chain_c_max_residue)
        
        chain_a_desc = f"{chain_a_start_residue}-end" if chain_a_start_residue else "full"
        print(f"Chain {chain_a_id} ({chain_a_desc}): {len(seq_a)} residues")
        print(f"Chain {chain_c_id} (1-{chain_c_max_residue}): {len(seq_c)} residues")
        
        # Ground truth for TM-score
        gt_coords = np.concatenate([coords_a, coords_c], axis=0)
        gt_seq = seq_a + seq_c
        
        len_a = len(seq_a)
        len_c = len(seq_c)
        linker_start = len_a
        linker_end = len_a + linker_length
        total_len = len_a + linker_length + len_c
        
        print(f"\nDesigning {linker_length}-residue linker...")
        print(f"Total sequence length: {total_len}")
        print(f"Linker positions: {linker_start+1}-{linker_end} (1-indexed)")
        print(f"Using {self.num_gpus} GPUs for parallel evaluation")
        
        # Initialize random linker
        current_linker = "".join([random.choice(AA_ALPHABET) for _ in range(linker_length)])
        current_seq = seq_a + current_linker + seq_c
        
        # Initial evaluation
        print("\nEvaluating initial sequence...")
        result = self.predict_structure(current_seq, gpu_id=0)
        
        linker_plddt = result['plddt'][linker_start:linker_end].mean()
        scaffold_coords = np.concatenate([
            result['ca_coords'][:len_a],
            result['ca_coords'][linker_end:]
        ], axis=0)
        scaffold_seq = current_seq[:len_a] + current_seq[linker_end:]
        
        try:
            tm_score = compute_tm_score(scaffold_coords, gt_coords, scaffold_seq, gt_seq)
        except:
            tm_score = 0.0
        
        current_score = plddt_weight * linker_plddt + tm_weight * tm_score
        
        best_linker = current_linker
        best_score = current_score
        best_plddt = linker_plddt
        best_tm = tm_score
        best_pdb = result['pdb_str']
        
        history = []
        temp = temperature
        
        pbar = tqdm(range(steps))
        for step in pbar:
            # Generate multiple candidates (one per GPU)
            candidates = []
            for i in range(self.num_gpus):
                num_muts = max(1, int(base_mutations * temp))  # More mutations at higher temp
                mutated_linker = self.mutate_linker(current_linker, num_mutations=num_muts)
                candidates.append(mutated_linker)
            
            # Evaluate all candidates in parallel on different GPUs
            import concurrent.futures
            
            def evaluate_candidate(args):
                idx, linker = args
                seq = seq_a + linker + seq_c
                return idx, self.predict_structure(seq, gpu_id=idx)
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_gpus) as executor:
                futures = [executor.submit(evaluate_candidate, (i, c)) for i, c in enumerate(candidates)]
                results = [f.result() for f in futures]
            
            # Find best candidate
            best_candidate_idx = -1
            best_candidate_score = -float('inf')
            best_candidate_result = None
            best_candidate_linker = None
            
            for idx, res in results:
                cand_linker = candidates[idx]
                cand_plddt = res['plddt'][linker_start:linker_end].mean()
                
                cand_scaffold_coords = np.concatenate([
                    res['ca_coords'][:len_a],
                    res['ca_coords'][linker_end:]
                ], axis=0)
                cand_scaffold_seq = seq_a + cand_linker + seq_c
                cand_scaffold_seq = cand_scaffold_seq[:len_a] + cand_scaffold_seq[linker_end:]
                
                try:
                    cand_tm = compute_tm_score(cand_scaffold_coords, gt_coords, cand_scaffold_seq, gt_seq)
                except:
                    cand_tm = 0.0
                
                cand_score = plddt_weight * cand_plddt + tm_weight * cand_tm
                
                if cand_score > best_candidate_score:
                    best_candidate_score = cand_score
                    best_candidate_idx = idx
                    best_candidate_result = res
                    best_candidate_linker = cand_linker
                    best_candidate_plddt = cand_plddt
                    best_candidate_tm = cand_tm
            
            # Metropolis acceptance
            delta = best_candidate_score - current_score
            if delta > 0 or random.random() < np.exp(delta / temp):
                current_linker = best_candidate_linker
                current_score = best_candidate_score
                linker_plddt = best_candidate_plddt
                tm_score = best_candidate_tm
                
                if best_candidate_score > best_score:
                    best_linker = best_candidate_linker
                    best_score = best_candidate_score
                    best_plddt = best_candidate_plddt
                    best_tm = best_candidate_tm
                    best_pdb = best_candidate_result['pdb_str']
            
            # Decay temperature
            temp *= temp_decay
            
            pbar.set_description(
                f"pLDDT: {linker_plddt:.3f} | TM: {tm_score:.3f} | Best: {best_plddt:.3f}/{best_tm:.3f} | T: {temp:.3f}"
            )
            
            # Save checkpoints
            if step % 10 == 0 or step == steps - 1:
                history.append({
                    "step": step,
                    "linker_plddt": float(linker_plddt),
                    "tm_score": float(tm_score),
                    "linker_seq": current_linker,
                    "temperature": float(temp)
                })
                
                with open(out_path / f"step_{step:04d}.pdb", "w") as f:
                    f.write(best_candidate_result['pdb_str'] if best_candidate_result else best_pdb)
        
        # Save final results
        final_seq = seq_a + best_linker + seq_c
        
        with open(out_path / "best.pdb", "w") as f:
            f.write(best_pdb)
        
        with open(out_path / "final_best.pdb", "w") as f:
            f.write(best_pdb)
        
        with open(out_path / "result.txt", "w") as f:
            f.write(f"Chain A sequence ({len_a} aa):\n{seq_a}\n\n")
            f.write(f"Designed linker ({linker_length} aa):\n{best_linker}\n\n")
            f.write(f"Chain C sequence ({len_c} aa):\n{seq_c}\n\n")
            f.write(f"Full sequence ({total_len} aa):\n{final_seq}\n\n")
            f.write(f"Final linker pLDDT: {best_plddt:.4f}\n")
            f.write(f"Final TM-score: {best_tm:.4f}\n")
        
        with open(out_path / "history.json", "w") as f:
            json.dump(history, f, indent=4)
        
        with open(out_path / "designed.fasta", "w") as f:
            f.write(f">designed_linker_plddt_{best_plddt:.4f}_tm_{best_tm:.4f}\n")
            f.write(f"{final_seq}\n")
        
        print(f"\n{'='*50}")
        print(f"Linker Design Complete!")
        print(f"{'='*50}")
        print(f"Designed linker: {best_linker}")
        print(f"Linker pLDDT: {best_plddt:.4f}")
        print(f"TM-score (scaffold): {best_tm:.4f}")
        print(f"Results saved to: {out_path}")
        
        return best_linker, final_seq


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU linker design using MCMC")
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB file")
    parser.add_argument("--chain_a", type=str, default="A", help="First chain ID")
    parser.add_argument("--chain_c", type=str, default="C", help="Second chain ID")
    parser.add_argument("--chain_a_start", type=int, default=None, help="Start residue for chain A")
    parser.add_argument("--chain_c_max", type=int, default=100, help="Max residue for chain C")
    parser.add_argument("--linker_len", type=int, default=30, help="Linker length")
    parser.add_argument("--steps", type=int, default=200, help="MCMC steps")
    parser.add_argument("--temperature", type=float, default=1.0, help="Initial temperature")
    parser.add_argument("--temp_decay", type=float, default=0.995, help="Temperature decay rate")
    parser.add_argument("--plddt_weight", type=float, default=1.0, help="pLDDT weight")
    parser.add_argument("--tm_weight", type=float, default=1.0, help="TM-score weight")
    parser.add_argument("--out_dir", type=str, default="linker_multigpu_results", help="Output directory")
    parser.add_argument("--weights", type=str, default="/home/ubuntu/esm_weights")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use")
    parser.add_argument("--mutations", type=int, default=3, help="Base number of mutations per step (higher = faster exploration)")
    
    args = parser.parse_args()
    
    designer = MultiGPULinkerDesigner(args.weights, num_gpus=args.num_gpus)
    designer.design_linker_mcmc(
        pdb_path=args.pdb,
        chain_a_id=args.chain_a,
        chain_c_id=args.chain_c,
        chain_a_start_residue=args.chain_a_start,
        chain_c_max_residue=args.chain_c_max,
        linker_length=args.linker_len,
        steps=args.steps,
        temperature=args.temperature,
        temp_decay=args.temp_decay,
        plddt_weight=args.plddt_weight,
        tm_weight=args.tm_weight,
        out_dir=args.out_dir,
        base_mutations=args.mutations
    )


if __name__ == "__main__":
    main()

