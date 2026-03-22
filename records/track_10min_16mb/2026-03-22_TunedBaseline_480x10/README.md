# Tuned Baseline: dim=480, 10 Layers

## Approach

Tuned hyperparameters on the baseline architecture. Key changes:

- **10 layers** (was 9) — more depth for better representations
- **dim=480** (was 512) — slightly narrower to fit more layers under 16MB
- **matrix_lr=0.06** (was 0.04) — faster convergence
- **scalar_lr=0.06** (was 0.04) — faster convergence for scalar params
- **warmdown_iters=1500** (was 1200) — longer decay for better final loss

Total params: ~16.6M (under 16MB compressed budget)

## Configuration

- `MODEL_DIM=480`
- `NUM_LAYERS=10`
- `NUM_KV_HEADS=4`
- `NUM_HEADS=8`
- `MLP_MULT=2`
- `MATRIX_LR=0.06`
- `SCALAR_LR=0.06`
- `WARMDOWN_ITERS=1500`

## Why This Should Beat Baseline

1. **More layers** (10 vs 9) = more depth = better representations
2. **Higher learning rates** = faster convergence in limited training time
3. **Longer warmdown** = smoother LR decay = better final loss

## Risks

- dim=480 vs 512 means slightly less width per layer
- May need to tune warmdown length
