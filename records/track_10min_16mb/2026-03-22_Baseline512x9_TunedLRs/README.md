# Baseline (dim=512, 9 layers) + Tuned Hyperparameters

## Approach

Keep the fast baseline architecture (dim=512, 9 layers) but tune the hyperparameters for better optimization:

- `matrix_lr`: 0.04 → **0.06** (50% increase for faster convergence)
- `scalar_lr`: 0.04 → **0.06** (50% increase)
- `tied_embed_lr`: 0.05 → **0.08** (60% increase)
- `warmdown_iters`: 1200 → **1500** (longer warmdown for better final convergence)

## Why This Should Beat the Previous Tuned Config

The previous tuned config (dim=480, 10 layers) was slower per step (~73ms vs ~43ms) because the additional layer adds sequential overhead that outweighs the width savings. This means fewer total steps (8,202 vs ~13,780) in the 10-minute window.

By keeping baseline architecture:
- ~43ms/step → ~13,780 steps in 10 min (vs 8,202)
- Combined with better LRs → should achieve ~1.22 val_bpb

## Technical Notes

- Uses manual `repeat_interleave` for KV heads instead of `enable_gqa=True` to avoid torch.compile fallback on Runpod 8xH100
- Fixed: `enable_gqa` with `torch.compile` causes severe performance regression on PyTorch 2.9.1

## Key Trade-off

Previous run (dim=480, 10L): val_bpb=1.2538, ~73ms/step, 8,202 steps
This run (dim=512, 9L): expected ~1.22 val_bpb, ~43ms/step, ~13,780 steps

Speed × steps wins over narrower/longer.
