# ESMFold Protein Hallucination & Linker Design

## Overview

This project implements protein sequence optimization using ESMFold, including:
1. **MCMC-based hallucination** - Monte Carlo optimization of sequences
2. **Gradient-based hallucination** - Backpropagation through soft embeddings
3. **Linker design** - Designing rigid linkers between protein chains

## Environment

```bash
conda env create -f env/mlfold.yml
conda activate mlfold
```

ESMFold weights: `/home/ubuntu/FORD/esmfold/esm_weights`

---

## Scripts

| Script | Description |
|--------|-------------|
| `hallucinate.py` | MCMC-based sequence optimization (simulated annealing) |
| `gd_hallucinate.py` | Gradient descent on sequence logits (frozen ESM weights) |
| `linker_design.py` | Design linkers between two protein chains |
| `animate_protein.py` | Create GIF animations from PDB snapshots |
| `make_trajectory.py` | Combine PDBs into multi-model trajectory |
| `view_pdb.py` | Browser-based 3D protein viewer |
| `view_protein.ipynb` | Jupyter notebook for interactive viewing |

---

## Experiments

### 1. MCMC Hallucination (`run1/`)

```bash
python hallucinate.py --length 100 --steps 500 --out_dir run1
```

- **Method**: Random mutations + Metropolis acceptance
- **Objective**: Maximize pLDDT
- **Result**: Basic sequence optimization, slower convergence

### 2. Gradient-Based Hallucination (`run2/`, `run2_test/`)

```bash
python gd_hallucinate.py --length 68 --steps 200 --lr 0.5 --out_dir run2
```

- **Method**: Optimize sequence logits via backpropagation
- **Key insight**: Freeze ESM weights, only update soft sequence representation
- **Result**: Faster convergence, smoother optimization landscape

---

## Linker Design Experiments

**Task**: Design a rigid linker connecting Chain A (C-terminal) to Chain C (N-terminal) of `design/design4.pdb`

**Objectives**:
1. Maximize linker pLDDT (confidence)
2. Maximize TM-score to ground truth (scaffold preservation)

### Memory Constraints (80GB A100)

| Total Length | Status |
|-------------|--------|
| 208 residues | ✓ Works |
| 228 residues | ✓ Works (limit) |
| 230 residues | ✗ OOM |

**Scaling formula**: `max_length ≈ 228 × sqrt(VRAM_GB / 80)`

| VRAM | Max Length | Per Chain (with 30 AA linker) |
|------|-----------|-------------------------------|
| 80GB | ~228 | 99 AA |
| 160GB | ~322 | 146 AA |
| 320GB | ~456 | 213 AA |
| 640GB | ~644 | 307 AA |

### Run 1 (`linker_run1/`) - Initial attempt with broken loss

```bash
python linker_design.py --pdb design/design4.pdb \
  --chain_a A --chain_c C \
  --chain_a_start 96 --chain_c_max 100 \
  --linker_len 8 --steps 100
```

**Issue**: `coord_loss` (~2000) dominated `plddt_loss` (~0.5)

```python
# Bad loss function
total_loss = plddt_weight * plddt_loss + tm_weight * coord_loss * 0.01
# coord_loss * 0.01 = 20.26 >> plddt_loss = 0.52
```

**Result**: pLDDT stuck at 0.52

### Run 3 (`linker_run3/`) - Fixed loss scaling ✓

```bash
python linker_design.py --pdb design/design4.pdb \
  --chain_a A --chain_c C \
  --chain_a_start 96 --chain_c_max 100 \
  --linker_len 8 --steps 100 \
  --plddt_weight 1.0 --tm_weight 1.0
```

**Fix**: Normalized RMSD to 0-1 scale

```python
# Fixed loss function
rmsd_loss_normalized = rmsd_loss / 50.0  # ~0-1 range
total_loss = plddt_weight * plddt_loss + tm_weight * rmsd_loss_normalized
```

| Metric | Value |
|--------|-------|
| Linker length | 8 AA |
| Designed linker | `AVERREQL` |
| Linker pLDDT | **0.66** |
| TM-score | 0.50 |
| Total residues | 208 |

### Run 4 (`linker_run4/`) - 30 AA Linker ✓ (Best)

```bash
python linker_design.py --pdb design/design4.pdb \
  --chain_a A --chain_c C \
  --chain_a_start 116 --chain_c_max 80 \
  --linker_len 30 --steps 150 \
  --plddt_weight 1.0 --tm_weight 1.0
```

| Metric | Value |
|--------|-------|
| Chain A | 80 AA (C-terminal) |
| Linker length | 30 AA |
| Chain C | 80 AA (N-terminal) |
| Designed linker | `AADWIIMLIKWFIIAIIEAIAIIAEIIAAF` |
| Linker pLDDT | **0.80** ⬆️ |
| TM-score | 0.52 |
| Total residues | 190 |

**Note**: The longer linker achieved much higher confidence. The hydrophobic sequence suggests a stable alpha helix.

---

## Key Learnings

### 1. Loss Function Scaling is Critical
When combining multiple objectives:
- Normalize all terms to similar scales (0-1 range)
- RMSD in Angstroms (~50Å) needs `/50` to match pLDDT (~0.5)

### 2. Memory Scales Quadratically
- Attention + pairwise representations: O(L²)
- For gradient-based optimization, need extra memory for backward pass
- 80GB GPU → max ~228 residues with gradients

### 3. Longer Linkers Can Be More Confident
- 8 AA linker: 0.66 pLDDT
- 30 AA linker: 0.80 pLDDT
- More residues give ESMFold more context to form stable secondary structure

---

## Usage for Multi-GPU (4× A800)

With ~320GB total VRAM, you could potentially run:

```bash
# Full Chain A (195 AA) + 30 AA linker + 100 AA Chain C = 325 residues
python linker_design.py --pdb design/design4.pdb \
  --chain_a A --chain_c C \
  --chain_a_start 1 --chain_c_max 100 \
  --linker_len 30 --steps 200 \
  --plddt_weight 1.0 --tm_weight 1.0 \
  --out_dir linker_full
```

**Note**: May need to implement model parallelism or gradient checkpointing for multi-GPU.

---

## Output Files

Each run produces:
- `best.pdb` - Best structure by linker pLDDT
- `final_best.pdb` - Same as best.pdb
- `step_XXXX.pdb` - Snapshots every 10 steps
- `history.json` - Optimization trajectory
- `result.txt` - Final summary
- `designed.fasta` - Final sequence in FASTA format

---

## Visualization

```bash
# Browser-based viewer
python view_pdb.py linker_run4/

# Create animation GIF
python animate_protein.py linker_run4/ --output animation.gif

# Jupyter notebook
jupyter notebook view_protein.ipynb
```

