# Reproducibility

This document gives the exact command sequence, pinned versions, expected runtimes, and expected score ranges to reproduce every published number in [`RESULTS.md`](RESULTS.md). Reported numbers are seed-deterministic where stated; cells that depend on stochastic sampling are flagged.

## 1. Pinned dependency stack

The training notebook [`notebooks/train_merchant_agent.ipynb`](../notebooks/train_merchant_agent.ipynb) installs and verifies the following pins. The setup cell asserts each pin and fails loud if any version slips.

| Package | Version | Why pinned |
|---|---|---|
| torch | 2.10.0+cu128 | Matches torchvision 0.25.0+cu128 and torchaudio 2.10.0+cu128 |
| transformers | 4.51.3 | TRL 0.21.0 was tested against 4.51.x; newer transformers break GRPOTrainer internals |
| trl | 0.21.0 | Provides `GRPOTrainer` with `reward_funcs` list and proper sampling kwarg passing |
| peft | 0.14.0 | Compatible with huggingface-hub 0.26.x; later peft requires hub 1.x |
| tokenizers | 0.21.4 | Required range for transformers 4.51.x |
| huggingface-hub | 0.26.5 | Last 0.x release; peft 0.14 imports paths that moved in hub 1.0 |
| accelerate | 1.0.1 | Compatible with hub 0.26; later accelerate hard-requires hub 1.x |
| openenv-core | ≥0.2.2 | Source of `Environment`, `Rubric`, `WeightedSum`, `Gate` |
| pydantic | ≥2.10 | Used by core models |
| datasets | ≥2.20,<4.0 | Compatible with the pinned transformers + tokenizers |

## 2. End-to-end reproduction (Colab T4)

### 2.1 Open the notebook

```
https://colab.research.google.com/github/MitudruDutta/ChargeBackOps/blob/main/notebooks/train_merchant_agent.ipynb
```

Connect to a T4 runtime (free tier suffices).

### 2.2 Run cells in order

Each cell is idempotent. Total wallclock ≈ 75 minutes on a free T4.

| Cell | Purpose | Wallclock | VRAM peak |
|---|---|---|---|
| `setup-code` | Pin install + repo clone + asserts | 4 min | 0 GB |
| `patch-code` | sys.path + module-cache flush | 5 sec | 0 GB |
| `model-code` | Load Qwen2.5-3B-Instruct fp16 + LoRA | 3 min | 6.3 GB |
| `sft-data-code` | Generate 4,000 SFT rows + chat-template wrap | 1 min | 6.3 GB |
| `sft-train-code` | Phase A SFT (150 steps) | 17 min | 8.4 GB |
| `merge-code` | Reload SFT, merge into base, attach Phase B LoRA | 2 min | 6.3 GB |
| `grpo-data-code` | Build GRPO state-action dataset with curriculum bias | 1 min | 6.3 GB |
| `grpo-train-code` | Phase B GRPO (200 steps) | 40 min | 11.5 GB |
| `eval-code` | Per-checkpoint eval + plot generation | 18 min | 12.5 GB |
| `diag-code` | Three-task diagnostic rollout | 2 min | 6.3 GB |

### 2.3 Override knobs

Set environment variables before running the relevant cell:

```python
import os
os.environ['MODEL_ID'] = 'Qwen/Qwen2.5-3B-Instruct'   # default
os.environ['SFT_TARGET_ROWS'] = '4000'                 # default
os.environ['SFT_MAX_STEPS'] = '150'                    # default
os.environ['SFT_LR'] = '1e-4'                          # default
os.environ['GRPO_MAX_STEPS'] = '200'                   # default
os.environ['GRPO_LR'] = '3e-5'                         # default
os.environ['PHASE_B_MAX_STATES_PER_TASK'] = '10'       # default
os.environ['GRPO_DIFFICULTIES'] = 'easy,medium,hard,nightmare'  # default
os.environ['RUN_SFT_TRAIN'] = 'auto'                   # auto-skip if adapter exists
os.environ['RUN_GRPO'] = '1'                           # set '0' to skip Phase B
```

## 3. End-to-end reproduction (local, ≥12 GB VRAM)

If you have a local machine with at least 12 GB CUDA VRAM, the same notebook runs unchanged. Adjust `WORK_DIR` at the top of the setup cell to a local writable path.

```bash
git clone https://github.com/MitudruDutta/ChargeBackOps
cd ChargeBackOps
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
jupyter notebook notebooks/train_merchant_agent.ipynb
```

For laptops with less VRAM, set `MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct` to fit in 8 GB. Expect lower absolute scores (the model is half the size) but the same qualitative training story.

## 4. Reproducing only the scripted-policy baseline sweep

No GPU required. Runs on CPU in under a minute.

```bash
pip install -e ".[dev]"
pytest -q tests/                           # 113 tests, all green
python -m runners.benchmark_runner         # prints headline + multi-seed sweep
```

Expected output (deterministic):

```
Headline catalog (12 tasks):
  naive          : 0.0000
  concede_all    : 0.4435
  escalate_all   : 0.7668
  heuristic      : 0.8132

Multi-seed grid (28 tasks):
  naive          : 0.0000
  concede_all    : 0.4454
  escalate_all   : 0.7675
  heuristic      : 0.7628

Marathon (long-horizon):
  naive          : 0.0000
  concede_all    : 0.4004
  escalate_all   : 0.6168
  heuristic      : 0.6793
```

These numbers are exact; the heuristic policy and arbitration adjudicator are deterministic given (case, packet).

## 5. Expected training-curve numbers (with seed variance)

The published training curve was produced with seeds:

```
SFT_SEED_START = 1000
HOLDOUT_SEEDS_BY_DIFF = {
    'easy':      {42},
    'medium':    {17, 99},
    'hard':      {7, 53},
    'nightmare': {31, 77},
}
```

Holdout seeds are excluded from training and used as the eval set.

Expected per-checkpoint scores (with ±0.03 absolute variance from sampling stochasticity in GRPO):

| Checkpoint | overall | easy | medium | hard | nightmare |
|---|---|---|---|---|---|
| Untrained base | 0.47 ± 0.02 | 0.29 ± 0.05 | 0.44 ± 0.04 | 0.77 ± 0.03 | 0.38 ± 0.05 |
| SFT | 0.75 ± 0.02 | 0.92 ± 0.04 | 0.79 ± 0.03 | 0.75 ± 0.04 | 0.55 ± 0.05 |
| GRPO | 0.73 ± 0.04 | 0.61 ± 0.08 | 0.79 ± 0.04 | 0.82 ± 0.05 | 0.69 ± 0.06 |

GRPO numbers have wider variance because the trainer's sampling is stochastic and only 30-50% of steps see a non-zero gradient (see [`METHOD.md`](METHOD.md) §3 for why).

## 6. Reproducing the figures

After the eval cell completes, two PNGs are written to `docs/figures/`:

- `training_curve.png` — overall mean score vs GRPO step, with heuristic and naive baselines as dashed lines.
- `training_curve_by_family.png` — per-difficulty curves on the same axes.

Both are committed to the repo so judges who do not run the notebook can still see the results.

## 7. Test suite

```bash
pytest -q tests/
```

Should output:

```
113 passed in ~7s
```

Failures here indicate environment, grader, or training-pipeline regressions. See `tests/conftest.py` for fixture details.

## 8. Running the trained agent

After the notebook completes, the SFT and GRPO adapters are saved under:

- `/content/sft-merchant-agent/final/` (or `WORK_DIR/sft-merchant-agent/final/` locally)
- `/content/grpo-merchant-agent/final/`

To use the trained model in the inference path:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-3B-Instruct', torch_dtype=torch.float16, device_map='cuda',
)
sft = PeftModel.from_pretrained(base, 'sft-merchant-agent/final')
merged = sft.merge_and_unload()
trained = PeftModel.from_pretrained(merged, 'grpo-merchant-agent/final')
trained.eval()

tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
# ... use trained as the policy in run_episode_with_text_policy()
```

See [`RUNNING_THE_AGENT.md`](RUNNING_THE_AGENT.md) for the full inference path.
