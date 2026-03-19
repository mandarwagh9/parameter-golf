# Run Guide: High-Risk Shared Pyramid Transformer

## Setup

### 1. Install Dependencies
```bash
pip install torch numpy sentencepiece huggingface-hub datasets tqdm
```

### 2. Download Data
```bash
cd parameter-golf
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
```

### 3. Smoke Test (Param Count)
```bash
python -c "
import torch, sys
sys.path.insert(0, 'records/track_10min_16mb/2026-03-19_SharedPyramid_T1408')
from train_gpt import SharedPyramidTransformer, Hyperparameters
args = Hyperparameters()
model = SharedPyramidTransformer(
    vocab_size=1024, model_dim=1408, num_heads=8, num_kv_heads=4,
    mlp_mult=2, tie_embeddings=True, tied_embed_init_std=0.005,
    logit_softcap=30.0, rope_base=10000.0, qk_gain_init=1.5,
    depth_recurrence=9
)
n_params = sum(p.numel() for p in model.parameters())
print(f'Params: {n_params:,} ({n_params/1e6:.2f}M)')
print(f'Expected: ~15,337,352')
# Estimate compressed size
import zlib, io
sd = {k: v.bfloat16() for k, v in model.state_dict().items()}
buf = io.BytesIO()
torch.save(sd, buf)
raw_bytes = len(buf.getvalue())
compressed = len(zlib.compress(buf.getvalue(), level=9))
print(f'Raw model bytes: {raw_bytes:,}')
print(f'Compressed (zlib9): {compressed:,} ({compressed/1e6:.2f}MB)')
print(f'Budget: 16,000,000 bytes')
print(f'Status: {\"PASS\" if compressed < 16000000 else \"FAIL\"}')"
```

### 4. Run Full Training (8xH100)
```bash
cd parameter-golf
torchrun --standalone --nproc_per_node=8 \
    records/track_10min_16mb/2026-03-19_SharedPyramid_T1408/train_gpt.py \
    --data_path=./data/datasets/fineweb10B_sp1024/ \
    --tokenizer_path=./data/tokenizers/fineweb_1024_bpe.model \
    --run_id=shared_pyramid_1408 \
    --model_dim=1408 \
    --depth_recurrence=9 \
    --val_loss_every=1000 \
    --train_log_every=200
```

### 5. Check Output
Look for `final_int8_zlib_roundtrip val_bpb:X.XXXX` in the logs.

---

## Architecture Summary

| Parameter | Baseline | This Submission |
|-----------|----------|-----------------|
| dim | 512 | **1408** |
| layers | 9 | **1 shared + 9 passes** |
| KV heads | 4 | 4 |
| vocab | 1024 | 1024 |
| params | 17.06M | **15.34M** |
| width ratio | 1x | **2.75x** |
| effective depth | 9 | **9** |

## Expected Outcome

**Best case:** val_bpb < 1.22 (beating baseline)
- Wider layers compensate for reduced parameter count
- Depth recurrence enables iterative refinement

**Expected:** val_bpb ~1.22-1.24
- Comparable to baseline, validating the approach

**Worst case:** val_bpb > 1.25
- Single shared block lacks layer diversity
- Fall back to wider standard architecture

## Fallback Configurations

If this doesn't work, try:

### Fallback 1: Standard Architecture, Tuned Hyperparams
```bash
# Just tune Muon LR and warmdown
MODEL_DIM=512 NUM_LAYERS=9 MATRIX_LR=0.06 WARMDOWN_ITERS=2000
```

### Fallback 2: Wider Standard Architecture
```bash
# Dim sweep within budget
MODEL_DIM=520 NUM_LAYERS=9
```

### Fallback 3: SwiGLU (dim=1312, T=1)
```bash
# Replace ReLU^2 with SwiGLU, keep wider dim
MODEL_DIM=1312 DEPTH_RECURRENCE=1
# Note: requires modifying train_gpt.py to swap MLP
```

## Troubleshooting

**CUDA OOM?** Reduce batch size:
```bash
TRAIN_BATCH_TOKENS=262144
```

**torch.compile fails?** Remove `fullgraph=True`:
```python
compiled_model = torch.compile(base_model, dynamic=False)
```

**Convergence issues?** Try lower Muon momentum:
```bash
MUON_MOMENTUM=0.9 MUON_BACKEND_STEPS=8
```
