# abstention-reasoning

Training language models to recognize when they cannot solve a problem and ask for a hint, rather than committing to an answer every time.

Models are trained in two stages: supervised fine-tuning (SFT) on chain-of-thought traces, then reinforcement learning with a reward function that pays for correct answers and charges for each hint taken. RL runs on a modified copy of [verl](https://github.com/volcengine/verl), vendored in `verl/`.

Everything is driven by one CLI:

```bash
python -m pipeline <command> [--task TASK] [--method METHOD] ...
```

## Install

Requires Python >= 3.10.12 and a CUDA GPU — vLLM backs all generation.

```bash
pip install -e .        # pipeline + deps (torch, transformers, vllm, trl, ...)
pip install -e verl/    # RL trainer
```

`verl/` is vendored in-tree, not a submodule. It carries local modifications (custom reward functions, multi-turn rollout hooks) and should not be swapped for an upstream checkout.

## Quickstart

The full `baseline` → `method_b` pipeline on `countdown` with a 1.5B student and Qwen3-14B as the teacher. Every path is written out: `--method` plus `--run-id` would derive the same ones, but spelling them makes the data flow between stages visible.

### 0. Problems — shared by every method

```bash
python -m pipeline create_primitives --task countdown --num-puzzles 5000 --seed 42 \
    --output data/countdown/problems/primitives.json

python -m pipeline create_partitions --task countdown --seed 42 \
    --primitives data/countdown/problems/primitives.json \
    --output data/countdown/problems
```

`create_partitions` writes one file per split (`sft_train`, `sft_val`, `rl_train`, `rl_val`, `eval`). Every method formats those same files, so the split is fixed once and recorded rather than recomputed per method.

### 1. Baseline Training

```bash
# Apply the baseline template to each partition
python -m pipeline create_prompts --task countdown --method baseline --split all \
    --data-name countdown

# Teacher rollouts -> SFT data
python -m pipeline generate --task countdown --method baseline --async \
    --model Qwen/Qwen3-14B --split sft_train \
    --data-name countdown

# SFT the student on those rollouts
python -m pipeline train_sft --task countdown --method baseline --run-id qwen2.5-1.5b \
    --base-model Qwen/Qwen2.5-1.5B \
    --data-name countdown

# RL from that SFT checkpoint -> the baseline model
python -m pipeline train_rl --task countdown --method baseline --run-id qwen2.5-1.5b \
    --data-name countdown

python -m pipeline evaluate --task countdown --method baseline --model rl --async \
    --run-id qwen2.5-1.5b --split eval \
    --data-name countdown
```

### 2. Method B Training

```bash
# Apply the method_b template to the same partitions
python -m pipeline create_prompts --task countdown --method method_b --split all \
    --data-name countdown

# Probe: how often does the teacher solve each problem unaided?
python -m pipeline generate --task countdown --method baseline --async \
    --model Qwen/Qwen3-14B --split sft_train --no-hints --num-samples 8 \
    --data-name countdown \
    --output data/countdown/.scratch/sft_train__baseline__probe.json

# Schedule hints from that probe -- harder problems get more -- and oversample
# until half the set is correct. The profile is derived in memory; only the
# dataset is written.
python -m pipeline generate --task countdown --method method_b --async \
    --model Qwen/Qwen3-14B --split sft_train \
    --hint-schedule data/countdown/.scratch/sft_train__baseline__probe.json \
    --hint-target-fraction 0.5 --max-hints 4 --target-correct-rate 0.5 \
    --data-name countdown

# Hint SFT on top of the baseline model
python -m pipeline train_sft --task countdown --method method_b --run-id qwen2.5-1.5b \
    --base-model models/countdown/baseline_rl/qwen2.5-1.5b/model \
    --data-name countdown

# RL with the quadratic hint penalty. alpha is the swept parameter and is not
# in the method config, so it must be passed here -- and it is the only thing
# distinguishing the runs, which is why it is in the run id.
python -m pipeline train_rl --task countdown --method method_b \
    --run-id qwen2.5-1.5b__quad-a0.5 \
    --sft-model models/countdown/method_b_sft/qwen2.5-1.5b/model \
    --reward-kwargs hint_penalty=0.1 hint_penalty_shape=quadratic hint_penalty_alpha=0.5 \
    --override algorithm.norm_adv_by_std_in_grpo=True \
    --data-name countdown

python -m pipeline evaluate --task countdown --method method_b --model rl --async \
    --run-id qwen2.5-1.5b__quad-a0.5 --split eval \
    --data-name countdown
```

Pass `--async` to `generate` and `evaluate` for the batched async vLLM path; it is the standard mode here. `--multi-turn` is read from the method config and does not need passing.

## Tasks

| Task | Problem |
|---|---|
| `countdown` | Reach a target number by combining given operands with `+ - * /` |
| `math` | Competition math problems (HuggingFace MATH) |
| `code_output` | Predict the stdout of a short program |

Each lives in `pipeline/tasks/{task}/` and implements the `BaseTask` interface: `create_primitives`, `format_prompt`, `check_correctness`.

## Methods

A *method* bundles a prompt-template variant, a reward function, and its RL settings into one named config (`pipeline/configs/methods/{task}/{method}.yaml`). The config name is also the name its artifacts carry on disk, so renaming a method renames its files. Four methods exist on countdown and competition math, three on code output (`method_b` is math-and-countdown only).

**Evaluated methods:**

| Method | Behavior |
|---|---|
| `baseline` | Answer directly, never ask for a hint. The starting point for every other method. |
| `hint_encourage` | Multi-turn: ask for hints, with a bonus for admitting a wrong answer. |
| `method_b` | Multi-turn hint-seeking inside a single `<think>` block, swept over the hint-penalty weight α. Countdown and competition math only. |

**SFT parents** — not evaluated on their own, but required to produce the hint methods above. Do not delete them:

| Method | Parent of |
|---|---|
| `hint` | `hint_encourage`, `method_b` |

Model sizes used throughout: Qwen2.5-1.5B, Qwen2.5-3B, Qwen3-4B.

Run `python -m pipeline list_methods --task <task>` to list what a task actually has.

Reward functions themselves live with the trainer, in `verl/recipe/{task}/reward_function.py`; a method config selects one by name and passes it `reward_kwargs`.

## Commands

**Discovery**

| Command | Purpose |
|---|---|
| `list_tasks` | List registered tasks |
| `list_methods` | List method configs for a task |

**Data**

| Command | Purpose |
|---|---|
| `create_primitives` | Generate raw puzzle data (shared across methods) |
| `create_partitions` | Split primitives into per-split problem files under `problems/` |
| `create_prompts` | Apply a method's template to a partition, or `all` of them |
| `create_ood_prompts` | Build OOD eval prompts (`aime2024`, `gsm8k`, `math500`, `minerva_math`, `olympiad_bench`, `unanswerable_math`) |

**Inference**

| Command | Purpose |
|---|---|
| `generate` | Run a model over prompts to produce a dataset (`--async`, `--no-hints`, `--hint-schedule`, `--target-correct-rate`) |
| `evaluate` | Evaluate a model (`--model sft`/`rl`, `--run-id`, `--num-samples`, `--async`) |
| `analyze` | Report accuracy by variant for a dataset or results file |

**Training**

| Command | Purpose |
|---|---|
| `train_sft` | SFT (`--epochs` 3, `--batch-size` 4, `--max-length` 4096) |
| `train_rl` | RL (`--total-steps` 400, `--train-batch-size` 64, `--save-freq` 25, `--reward-kwargs`, `--override`) |
| `convert_checkpoint` | Convert an FSDP/Megatron checkpoint to HuggingFace format |

`python -m pipeline <command> --help` documents every flag.

Every command that touches a model path requires `--run-id`; nothing is derived from the base checkpoint. Commands that read or write artifacts take an explicit `--primitives` / `--prompts` / `--dataset` / `--output`, and fall back to the method-derived path when omitted.

## Data model

Each stage is a pure transformation that writes new files and never edits existing ones:

```
problems/primitives.json                      Raw puzzle data (index, variant, task fields)
       │
       │  create_partitions
       ▼
problems/{split}.json                         The seeded split, written down once
       │
       │  create_prompts        + a method's template
       ▼
problems_with_format/{split}__{method}.json   Model-ready inputs + ground truth
       │                        .parquet for rl_train / rl_val
       │  generate              + a model's rollouts
       ▼
sft_datasets/{split}__{method}.json           Generations + correctness labels
       │
       │  train_sft -> train_rl
       ▼
models/{method}_{sft,models}/{run_id}/model/
       │
       │  evaluate
       ▼
models/{method}_{sft,models}/{run_id}/evals/{split}.json
```

### Splits

Splits are disjoint slices of a seeded shuffle of the primitives, so no problem appears in both training and evaluation. **Tasks declare their own layout**, and not every task defines every split:

| Split | `countdown`, `math` | `code_output` |
|---|---|---|
| `sft_train` | 0–27% | 0–17.28% |
| `sft_val` | 27–30% | 17.28–19.2% |
| `rl_train` | 30–65% | 19.2–67.2% |
| `rl_val` | 65–70% | — |
| `eval` | 70–100% | 67.2–100% |

Materialize whichever of `sft_train`/`sft_val` a stage needs — train on `sft_train`, and (optionally) compute validation loss during that training on `sft_val`.

`code_output` allocates its whole range across three regions, so it has no `rl_val`.

`--split all` creates exactly the splits the task defines — a task's `SPLITS` table in `pipeline/tasks/{task}/task.py` is the single source of truth.

## Artifacts

Generated data and model weights are written under two separate roots, each keyed by its own name and one tree per `{data_name}`/`{models_name}`:

* `data/{data_name}/` — problems, formatted prompts, and SFT datasets. Small, text, meant to be committed to git.
* `models/{models_name}/` — model weight checkpoints. Large, binary, meant to be pushed to a Hugging Face hub (see `upload_tools/`), not git — `models/` is gitignored.

`data_name` defaults to `--task` and `models_name` defaults to `data_name`; both can be overridden independently with `--data-name`/`--models-name`. For example `--task math --data-name math_o1` reads/writes under `data/math_o1/` and, unless `--models-name` is also given, `models/math_o1/`.

Three data layers under `data/{data_name}/`, distinguished by what is *in* a file rather than by which stage reads it:

* `problems/` — problems, no template.
* `problems_with_format/` — a partition with a method's template applied. Model-ready input, nothing generated yet.
* `sft_datasets/` — the above plus generations (teacher rollouts + correctness labels).

The RL parquet lives in `problems_with_format/`, not `sft_datasets/`: verl produces its own rollouts, so the file holds prompts and ground truth and no generations. Eval prompts sit there for the same reason; eval *results* go inside the run that produced them, under `models/{models_name}/`.

Names are `{partition}__{method}` and `{partition}__{method}__{optional_desc}`. Every field is separated by `__`, because method names contain single underscores (`method_b`, `method_ac`) and a single-underscore desc separator would make `sft_train__method_b_generations` ambiguous. Hyphens are legal inside a field — that is what carries model slugs (`qwen3-4b-base`) and run descs (`quad-a0.5`).

```yaml
data/{data_name}/
    problems/   # raw (question, answer, list of hint) partitions

        primitives.json
        sft_train.json
        sft_val.json                # needed; the predictor models are trained by SFT only
        rl_train.json
        rl_val.json
        eval.json

    problems_with_format/   # partition + template + hints. no generations.

        sft_train__baseline.json
        sft_val__baseline.json
        rl_train__baseline.parquet
        rl_val__baseline.parquet
        eval__baseline.json

        sft_train__method_b.json
        sft_val__method_b.json
        rl_train__method_b.parquet
        rl_val__method_b.parquet
        eval__method_b.json

        sft_train__method_ac.json   # A and C share one generation model
        sft_val__method_ac.json
        rl_train__method_ac.parquet
        rl_val__method_ac.parquet
        eval__method_ac.json

        sft_train__method_a__qh_predictions.json    # predictors: SFT only, no RL
        sft_val__method_a__qh_predictions.json
        sft_train__method_c__qhr_predictions.json
        sft_val__method_c__qhr_predictions.json

    sft_datasets/   # + teacher rollouts and correctness labels.
        sft_train__baseline.json
        sft_val__baseline.json

        sft_train__method_b.json
        sft_val__method_b.json

        sft_train__method_ac.json
        sft_val__method_ac.json
        sft_train__method_ac__generations.internal.json  # not used for training. self-generated
        sft_val__method_ac__generations.internal.json

        sft_train__method_a__qh_predictions.json
        sft_val__method_a__qh_predictions.json
        sft_train__method_c__qhr_predictions.json
        sft_val__method_c__qhr_predictions.json

    .scratch/   # intermediates read back by a later step, never trained on
        sft_train__baseline__probe.json

models/{models_name}/     # {method}_sft = SFT intermediate, {method}_rl = finished model, post-RL.
                           # {method}_predictors = SFT-only predictor (method_a only).
        baseline_sft/
            qwen2.5-1.5b/
                model/              # contains the hugging face.
                evals/              # goes into the actual model folder not decoupled
            qwen2.5-3b/
            qwen3-4b-base/
            qwen3-4b/
        baseline_rl/
            qwen2.5-1.5b/
            qwen2.5-3b/
            qwen3-4b-base/
            qwen3-4b-base__s2/      # seed replicate
            qwen3-4b/
        method_b_sft/
            qwen2.5-1.5b/           # SFT on top of baseline_rl
            qwen2.5-3b/
            qwen3-4b-base/
        method_b_rl/
            qwen2.5-1.5b__quad-a0.5/
            qwen2.5-3b__quad-a0.5/
            qwen3-4b-base__quad-a0.5/
        method_ac_sft/
            qwen2.5-1.5b/
            qwen2.5-3b/
            qwen3-4b-base/
            qwen3-4b/
        method_ac_rl/
            qwen2.5-1.5b/
            qwen2.5-3b/
            qwen3-4b-base/
            qwen3-4b/
        method_a_predictors/       # SFT only, warm-started from method_ac_rl
            qwen2.5-1.5b/
            qwen2.5-3b/
            qwen3-4b-base/
            qwen3-4b/
        method_c_predictors_sft/   # SFT warm-start for the method_c verifier
            qwen2.5-1.5b/
            qwen2.5-3b/
            qwen3-4b-base/
            qwen3-4b/
        method_c_rl/               # RL-trained method_c verifier
            qwen2.5-1.5b/
            qwen2.5-3b/
            qwen3-4b-base/
            qwen3-4b/
```

`method_ac` is the fixed-hint solver shared by Method A and C, trained the ordinary SFT+RL way. `method_a` and `method_c` are their respective verifiers (`method_a` predicts "can this be solved given the hints so far?", `method_c` checks a candidate solution): `method_a` is SFT-only, warm-started from `method_ac_rl` and landing straight in `models/{models_name}/method_a_predictors/`; `method_c` also gets an RL stage on top of its SFT warm-start (`models/{models_name}/method_c_predictors_sft/` -> `models/{models_name}/method_c_rl/`, the latter auto-derived like any other method).

Names come from the method's config filename, so `baseline.yaml` produces `data/{data_name}/problems_with_format/sft_train__baseline.json` and `models/{models_name}/baseline_sft/`.

`--run-id` names the run directory verbatim and is **required** wherever a model path is derived. A run is `{model_slug}` or `{model_slug}__{desc}`, where `{desc}` marks a deviation from the default recipe — so a default run is the bare slug (`qwen2.5-1.5b`) and only the varying part is named (`qwen2.5-1.5b__quad-a0.5`, `qwen3-4b-base__s2`). Nothing is inferred from the base checkpoint; a name guessed in code would only drift from the convention.

## Repository layout

```
pipeline/
├── __main__.py              # CLI entrypoint
├── commands/                # data, inference, training
├── configs/methods/{task}/  # method configs
├── core/                    # io, generator (vLLM), method paths
└── tasks/{task}/            # task class + prompt templates
verl/                        # modified verl trainer + reward functions
```

## Adding a task

1. Create `pipeline/tasks/{task}/` with a class extending `BaseTask`, implementing `create_primitives`, `format_prompt`, and `check_correctness`.
2. Override `SPLITS` if the default split layout does not suit the data.
3. Add templates under `pipeline/tasks/{task}/templates/{variant}/`.
4. Add method configs under `pipeline/configs/methods/{task}/`.
5. Register the class in `pipeline/tasks/__init__.py`.
