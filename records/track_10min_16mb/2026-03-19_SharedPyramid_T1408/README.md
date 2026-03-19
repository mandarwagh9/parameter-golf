# Shared Pyramid Transformer with Depth Recurrence

## Approach

This submission introduces **two key architectural innovations** over the baseline:

### 1. Shared Block Weight Tying

Instead of storing 9 independent transformer blocks (each ~1.9M params), this model stores **a single shared block** reused across 9 depth-recurrent passes.

| Component | Baseline | This Submission |
|-----------|----------|----------------|
| Transformer blocks | 9 separate blocks (~17M params) | 1 shared block (~13.9M params) |
| Width (dim) | 512 | **1408** (2.75x wider) |
| Total params | 17.06M | 15.34M |

The shared block is used sequentially 9 times (T=9), giving the model an **effective depth of 9** — same as baseline — while dramatically reducing parameter count.

This is inspired by:
- **Universal Transformers** (Dehghani et al., 2019) — recurrent transformer layers
- **Weight Tying** (Press & Wolf, 2016) — sharing across layers
- **Evolved Transformer** (So et al., 2019) — discovering efficient weight sharing

### 2. Width vs Depth Tradeoff

The saved parameters from weight sharing are reinvested into **2.75x wider layers**:

- Wider Q/K/V projections: more representational capacity per token
- Wider MLP: richer hidden representations
- Same effective depth (9 recurrent passes)

This exploits the intuition that **width** may be more important than **diversity of layers** in the parameter-constrained regime.

## Architecture

```
Token Embedding (tied, 1024 x 1408)
  └─→ RMSNorm
       └─→ for t in [1..9]:          # Depth recurrence
            ├─→ Skip-weighted input (learned skip_weights[t])
            ├─→ SharedBlock:
            │   ├─→ RMSNorm(Q/K pre-RoPE)
            │   ├─→ RoPE (base=10000, head_dim=176)
            │   ├─→ GQA (8 Q heads / 4 KV heads)
            │   ├─→ ReLU² MLP (1408 → 2816 → 1408)
            │   └─→ Learned residual gating (attn_scale, mlp_scale)
            └─→ Store in skip list
       └─→ Final RMSNorm → Tied LM Head
```

**Key specs:**
- `dim=1408`, `num_heads=8`, `num_kv_heads=4`, `mlp_mult=2`
- `depth_recurrence=9` (one shared block, 9 passes)
- `tie_embeddings=True`, `logit_softcap=30.0`, `qk_gain_init=1.5`
- Total: **15,337,352 params**

## Why This Should Win

1. **Same compressed budget, much wider representation**: 2.75x wider Q/K/V/MLP layers mean each token gets processed with more parameters per pass.

2. **Depth recurrence enables iterative refinement**: The model can refine its representations across multiple passes through the shared block, similar to how humans re-read complex sentences.

3. **Reduced parameter count with preserved expressivity**: By sharing one block instead of nine, we save ~1.7M parameters that can be reinvested into width.

4. **Same training recipe**: Muon optimizer, warmdown schedule, GQA, and all hyperparameters remain identical to baseline — isolating the architecture as the variable.

## Risks

- **Single shared block may lack layer diversity**: In standard transformers, different layers specialize (early layers capture syntax, later layers capture semantics). A shared block may not develop this specialization.
- **Depth recurrence convergence**: The model must learn to effectively use multiple passes. Insufficient training tokens may leave the model undertrained.
- **Optimization dynamics**: Muon + depth recurrence may interact differently than expected.

## Submission

- Architecture: Shared Pyramid Transformer with Depth Recurrence
- Config: `MODEL_DIM=1408 DEPTH_RECURRENCE=9`
- Target: Beat 1.2244 val_bpb on 10-min 8xH100 leaderboard
- Compute: Runpod 8xH100 SXM, ~10 minute wallclock cap
