# ifdr — Influence Function Driven Reasoning

## Repository layout

| Path | Description |
|------|-------------|
| `run_sft.py`, `run_sft_sweep.py`, `run_sft.sh` | Supervised fine-tuning entry points |
| `run_dpo.py` | DPO training |
| `run_rm.py`, `run_rm_bstn.py`, `run_rm_sweep.py`, `eval_rm.py` | Reward model training / evaluation |
| `run_gen_sp.py` | Generation |
| `eva_llm.py`, `eva_llm_sweep.py` | LLM evaluation |
| `evol_data_merge.py` | Data merging utilities |
| `openrlhf/` | Core RLHF library (datasets, models, trainers, utils) |
| `evol/` | Evolution pipeline (mutation/critic prompts, fitness, runners) |
| `eval/` | Math evaluation harness (incl. `latex2sympy`, benchmark data under `eval/data/`) |
| `analys/` | Analysis scripts |
| `scripts/` | Shell launch scripts |
| `utils/` | Shared helpers (metrics, prompts, logging) |
| `srl_math_benchmark/` | Math benchmark question banks (jsonl) |

## Not included in this repository

The following are excluded (see `.gitignore`) and must be provided separately:

- **Model checkpoints / weights** — `checkpoints/`, `*.pt`, `*.safetensors`, etc.
- **Training datasets** — `data_new/`, `*.pkl`
- **Training outputs** — the `wz/` output directory and per-experiment result JSON/JSONL files
- **vLLM** — install from source or PyPI (`pip install vllm`); the repo previously vendored a full clone

## Setup

```bash
pip install -r requirements.txt
```
