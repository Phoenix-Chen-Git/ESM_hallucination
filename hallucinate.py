import os
import torch
import numpy as np
import argparse
import time
import json
from pathlib import Path
from transformers import AutoTokenizer, EsmForProteinFolding
from tqdm import tqdm

# Constants
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

class ESMHallucinator:
    def __init__(self, model_path, device=None):
        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = EsmForProteinFolding.from_pretrained(model_path, weights_only=False)
        
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded on {self.device}")

    def fold(self, sequence):
        inputs = self.tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
        inputs = {key: val.to(self.device) for key, val in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        plddt = outputs.plddt[0].cpu().numpy()
        pdb_str = self.model.output_to_pdb(outputs)[0]
        
        return {
            "plddt": plddt,
            "mean_plddt": float(plddt.mean()),
            "pdb": pdb_str
        }

def mutate_sequence(seq, num_mutations=1, fixed_mask=None):
    seq_list = list(seq)
    indices = [i for i in range(len(seq_list)) if not (fixed_mask and fixed_mask[i])]
    
    if not indices:
        return seq
    
    to_mutate = np.random.choice(indices, min(num_mutations, len(indices)), replace=False)
    for idx in to_mutate:
        current_aa = seq_list[idx]
        possible_aas = [aa for aa in AA_ALPHABET if aa != current_aa]
        seq_list[idx] = np.random.choice(possible_aas)
        
    return "".join(seq_list)

def main():
    parser = argparse.ArgumentParser(description="ESMFold Hallucination (MCMC)")
    parser.add_argument("--length", type=int, default=100, help="Length of sequence to hallucinate")
    parser.add_argument("--steps", type=int, default=1000, help="Number of MCMC steps")
    parser.add_argument("--start_temp", type=float, default=0.01, help="Starting temperature for MCMC")
    parser.add_argument("--end_temp", type=float, default=0.0001, help="Ending temperature for MCMC")
    parser.add_argument("--mut_per_step", type=int, default=1, help="Number of mutations per step")
    parser.add_argument("--out_dir", type=str, default="hallucination_results", help="Output directory")
    parser.add_argument("--weights", type=str, default="/home/ubuntu/esm_weights", help="Path to ESMFold weights")
    parser.add_argument("--seed_seq", type=str, default=None, help="Starting sequence")
    parser.add_argument("--fixed_pos", type=str, default=None, help="Fixed positions (1-indexed, comma-separated)")
    
    args = parser.parse_args()
    
    # Setup output directory
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize hallucinator
    hallucinator = ESMHallucinator(args.weights)
    
    # Initialize sequence
    if args.seed_seq:
        current_seq = args.seed_seq
        args.length = len(current_seq)
    else:
        current_seq = "".join(np.random.choice(list(AA_ALPHABET), args.length))
        
    # Fixed positions
    fixed_mask = [False] * args.length
    if args.fixed_pos:
        for p in args.fixed_pos.split(","):
            try:
                if "-" in p:
                    start, end = map(int, p.split("-"))
                    for i in range(start-1, end):
                        if 0 <= i < args.length:
                            fixed_mask[i] = True
                else:
                    idx = int(p) - 1
                    if 0 <= idx < args.length:
                        fixed_mask[idx] = True
            except ValueError:
                print(f"Warning: could not parse fixed position {p}")

    # Initial fold
    print(f"Starting sequence: {current_seq}")
    result = hallucinator.fold(current_seq)
    current_score = result["mean_plddt"]
    best_score = current_score
    best_seq = current_seq
    
    print(f"Initial pLDDT: {current_score:.4f}")
    
    history = []
    csv_path = out_path / "progress.csv"
    with open(csv_path, "w") as f:
        f.write("step,plddt,best_plddt,temp,accepted\n")
    
    # Optimization loop
    pbar = tqdm(range(args.steps))
    for i in pbar:
        # Schedule temperature
        curr_temp = args.start_temp * (args.end_temp / args.start_temp) ** (i / args.steps)
        
        # Mutate
        new_seq = mutate_sequence(current_seq, num_mutations=args.mut_per_step, fixed_mask=fixed_mask)
        
        # Fold
        new_result = hallucinator.fold(new_seq)
        new_score = new_result["mean_plddt"]
        
        # Accept/Reject (Metropolis)
        delta = new_score - current_score
        accept = False
        if delta > 0:
            accept = True
        else:
            prob = np.exp(delta / curr_temp)
            if np.random.random() < prob:
                accept = True
                
        if accept:
            current_seq = new_seq
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_seq = current_seq
                # Save best PDB
                with open(out_path / "best.pdb", "w") as f:
                    f.write(new_result["pdb"])
                with open(out_path / "best_seq.txt", "w") as f:
                    f.write(f">step_{i}_plddt_{best_score:.4f}\n{best_seq}\n")
        
        # Logging
        pbar.set_description(f"Best: {best_score:.4f} Curr: {current_score:.4f} T: {curr_temp:.2e}")
        
        with open(csv_path, "a") as f:
            f.write(f"{i},{current_score:.4f},{best_score:.4f},{curr_temp:.2e},{1 if accept else 0}\n")
        
        if i % 100 == 0:
            history.append({
                "step": i,
                "score": current_score,
                "best_score": best_score,
                "seq": current_seq
            })
            # Save periodic PDB
            with open(out_path / f"step_{i:04d}.pdb", "w") as f:
                f.write(new_result["pdb"])

    # Final save
    final_result = hallucinator.fold(best_seq)
    with open(out_path / "final_best.pdb", "w") as f:
        f.write(final_result["pdb"])
        
    with open(out_path / "history.json", "w") as f:
        json.dump(history, f, indent=4)
        
    print(f"\nDone! Best pLDDT: {best_score:.4f}")
    print(f"Results saved to {args.out_dir}")

if __name__ == "__main__":
    main()

