# CoTEvol

(https://arxiv.org/abs/2604.14768)

## Abstract

Large Language Models (LLMs) exhibit strong mathematical reasoning when trained on high-quality Chain-of-Thought (CoT) that articulates intermediate steps, yet costly CoT curation hinders further progress. While existing remedies such as distillation from stronger LLMs and self-synthesis based on test-time search alleviate this issue, they often suffer from diminishing returns or high computing overhead. In this work, we propose COTEVOL, a genetic evolutionary framework that casts CoT generation as a population-based search over reasoning trajectories. Candidate trajectories are iteratively evolved through reflective global crossover at the trajectory level and local mutation guided by uncertainty at the step level, enabling holistic recombination and fine-grained refinement. Lightweight, task-aware fitness functions are designed to guide the evolutionary process toward accurate and diverse reasoning. Empirically, COTEVOL improves correct-CoT synthesis success by over 30% and enhances structural diversity, with markedly improved efficiency. LLMs trained on these evolutionary CoT data achieve an average gain of 6.6% across eight math benchmarks, outperforming previous distillation and self-synthesis approaches. These results un-
derscore the promise of evolutionary CoT synthesis as a scalable and effective method for
mathematical reasoning tasks.


## Method

<p align="center">
  <img src="assets/method.png" alt="CoTEvol framework overview" width="90%">
</p>

> **TODO:** Put your method/framework figure at `assets/method.png`
> (see `assets/README.md`). Add a one- or two-sentence caption describing the pipeline.

## Installation

It is recommended that the Python version for the operating environment be **3.10+**,
and we suggest using **PyTorch 2.5+**.

```bash
cd cotevol
pip install -r requirements.txt
```

## CoTs Synthesis

Please run this command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python ./evol/run_evolu_batch.py --config_path ./evol/evoluation+.yaml
```

You can replace the file paths and hyperparameters in `evoluation+.yaml`.

## SFT for Mathematical Reasoning Using Evolutionary CoTs

Please run this command:

```bash
python run_sft_sweep.py 0,1,2,3 qwen2.5-7b-it s1k_evol sft
```

## Evaluation

Please run this command:

```bash
bash ./scripts/eva_llm_sweep.sh qwen2.5-7b-it 01 trained_model_dir
```

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

## Environment variables

Some scripts read API credentials from environment variables instead of hardcoding them:

- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT` — for Azure OpenAI calls (`analys/reasonflux.py`)
- `WANDB_API_KEY` — for Weights & Biases logging
