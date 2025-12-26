# ESMFold Gradient-Based Linker Design: Lessons Learned

## The Challenge

We wanted to design a **30-residue linker** between:
- **Chain A**: 195 amino acids (full chain)
- **Chain C**: 100 amino acids (residues 1-100)
- **Total sequence length**: 325 residues

The goal was to use **gradient descent** (not MCMC) to optimize the linker sequence by backpropagating through ESMFold.

---

## Why ESMFold is Memory-Hungry

ESMFold's architecture creates massive memory requirements during backpropagation:

1. **ESM-2 Encoder** (33 layers): Standard transformer with O(L²) attention
2. **Structure Module (Trunk)** (48 blocks): Each block has:
   - Pairwise representation: `(B, L, L, 128)` tensor
   - Triangular attention: O(L²) operations
   - Multiple attention heads storing intermediate activations

For L=325 residues:
- Pairwise tensors alone: 325 × 325 × 128 × 4 bytes = **54 MB per tensor**
- With 48 blocks and multiple heads: **tens of GB** just for activations
- Backward pass needs to store all these for gradient computation

---

## What We Tried (and Failed)

### 1. Standard Gradient Descent
**Result**: CUDA OOM immediately
- Standard backprop stores all intermediate activations
- 80GB GPU not enough for 325-residue sequence

### 2. Gradient Checkpointing on ESM Encoder Only
```python
self.model.esm.encoder.gradient_checkpointing = True
```
**Result**: Still OOM
- Only checkpoints the ESM-2 encoder
- Trunk (structure module) still stores all activations

### 3. Mixed Precision (FP16)
```python
with torch.cuda.amp.autocast():
    output = self.forward_soft(full_logits)
```
**Result**: Still OOM
- Halves memory for forward pass
- But gradient computation still needs too much memory
- Also broke SVD operation (had to cast back to float32 for Kabsch RMSD)

### 4. Reducing Recycles
```python
self.model.trunk.no_recycles = 0
```
**Result**: Still OOM
- Fewer recycles = fewer forward passes
- But a single forward+backward still exceeds memory

### 5. FSDP (Fully Sharded Data Parallel) with 4 GPUs
```python
self.model = FSDP(self.model, sharding_strategy=ShardingStrategy.FULL_SHARD)
```
**Result**: Still OOM
- FSDP shards model weights across GPUs
- But doesn't help with activation memory during backprop
- Also had dtype compatibility issues (ESMFold has mixed float16/float32 params)

---

## The Solution That Worked ✅

### Aggressive Per-Block Gradient Checkpointing

The key insight: **Checkpoint each individual block in the trunk**, not just the whole trunk or encoder.

```python
from torch.utils.checkpoint import checkpoint

# Monkey-patch each trunk block to use gradient checkpointing
def block_forward_with_checkpoint(self_block, *args, **kwargs):
    return checkpoint(
        self_block.__class__.forward, 
        self_block, 
        *args, 
        **kwargs, 
        use_reentrant=False
    )

for block in self.model.trunk.blocks:
    block.forward = block_forward_with_checkpoint.__get__(block, block.__class__)
```

### Why This Works

1. **Checkpointing trades compute for memory**: Instead of storing activations, recompute them during backward pass
2. **Per-block granularity is key**: Checkpointing the whole trunk at once doesn't help because inputs to the trunk are still huge. Per-block checkpointing only keeps one block's activations at a time.
3. **Combined with other optimizations**:
   - Small chunk size: `self.model.trunk.chunk_size = 4`
   - FP16 forward pass: `torch.cuda.amp.autocast()`
   - Zero recycles: `no_recycles=0`

### Full Working Configuration

```python
# 1. Enable gradient checkpointing on ESM encoder
self.model.esm.encoder.gradient_checkpointing = True

# 2. Reduce trunk chunk size
self.model.trunk.chunk_size = 4

# 3. Monkey-patch EVERY trunk block with checkpointing
for block in self.model.trunk.blocks:
    block.forward = block_forward_with_checkpoint.__get__(block, block.__class__)

# 4. Use mixed precision for forward pass
with torch.cuda.amp.autocast():
    output = self.forward_soft(full_logits, num_recycles=0)

# 5. Move RMSD computation outside autocast (SVD doesn't support fp16)
rmsd_a = self.kabsch_rmsd(pred_a.float(), gt_coords_a.float())
```

---

## Results

| Metric | Value |
|--------|-------|
| Sequence Length | 325 residues |
| GPU Memory Used | ~70-75 GB (single A800) |
| Time per Step | ~30 seconds |
| Starting pLDDT | 0.588 |
| Final pLDDT (10 steps) | 0.715 |
| Improvement | +22% |

---

## Key Takeaways

1. **Gradient checkpointing granularity matters**: Per-block checkpointing >> whole-model checkpointing
2. **ESMFold's trunk is the bottleneck**: The structure module with its pairwise representations is what kills memory
3. **Mixed precision helps but isn't enough**: FP16 alone doesn't solve the activation memory problem
4. **FSDP doesn't magically solve everything**: It shards weights, not activations
5. **Always test with small steps first**: We used 10 steps to validate before running longer optimizations

---

## Files

- `linker_design_gradient_multigpu.py`: Working gradient-based linker design script
- `linker_fullA_30aa_gd/`: Output directory with results

---

## Hardware Used

- 4× NVIDIA A800-SXM4-80GB (320GB total)
- Only 1 GPU needed with per-block checkpointing
- Other 3 GPUs can run parallel experiments with different seeds

