# Danish Railway Journey RLVR — Tinker Cookbook

Fine-tune a language model to plan Danish train journeys using **Reinforcement Learning with Verifiable Rewards (RLVR)**.

The reward is not human preference — it is computed deterministically from:
- parsed arrival time and transfer count
- route plausibility (chronology + minimum skiftetid)
- optional re-verification via the RAPTOR router in `dsb_routing.py`

This follows the [Tinker Cookbook](https://github.com/thinking-machines-lab/tinker-cookbook) `math_rl` pattern.

## Pipeline overview

```
all_services.jsonl          journeys.jsonl
 (layer 1: parser)    ->   (layer 2: router facit)
        |                          |
        +----------+---------------+
                   v
            tinker_dsb_rl/
              journey_env.py      <- RL environment
              journey_grading.py  <- verifiable reward
              train.py            <- GRPO / importance_sampling RL
              train_sft.py        <- optional warm-start
```

## RAPTOR (ground-truth router)

**RAPTOR** = **R**ound-b**A**sed **P**ublic **T**ransit **O**ptimized **R**outer.

It finds earliest-arrival journeys on a public-transport timetable by working in **rounds** (each round = take one more vehicle / transfer), instead of running classic street-style Dijkstra on every stop edge.

In this project, RAPTOR lives in `dsb_routing.py` as `Timetable.earliest_journey(...)`. It is used to:

1. **Generate facit** — ground-truth journeys in `journeys*.jsonl`
2. **Verify rewards** — the `router` component in RLVR checks whether the model matches the RAPTOR optimum
3. **Stay as source of truth** — the LLM selects from retrieved services; RAPTOR (not the LLM) defines correctness for optimal arrival

Experiments A/B still use RAPTOR under the hood; the fine-tuned model learns to reason over a **local** timetable snippet, not to replace RAPTOR.

## Curriculum training

The successful runs do not train one hard routing task from scratch. They follow a
**curriculum**: start from a small, fully supervised format, then add one difficulty
at a time, always warm-starting from the previous checkpoint.

```text
Base Qwen3.5-4B
  → format SFT          (journey-shaped text)
  → A: 0 transfers      (SFT → RLVR)
  → B: 1 transfer       (SFT → RLVR)
  → C: 2 transfers      (SFT → RLVR)
       ├→ D: rank candidates     (SFT only; isolate optimality)
       └→ E: expanded subgraph   (SFT → RLVR; gold-free retrieval)
            └→ F: reasoning traces (SFT → RLVR; think then answer)
```

Each stage reuses the previous **weights**, not just the code. The new dataset
teaches the next skill; RLVR then optimizes the verifiable reward on that skill.

### Why a curriculum

Without retrieval and with mixed multi-transfer data, RLVR was stuck: `format≈1`
but `route=0` and `correct=0`. The model could write a journey, but almost never
a real service chain, so group-relative RL had no useful advantage signal.

The curriculum fixes that in three ways:

1. **Retrieval first.** The model reads a local timetable snippet instead of
   memorizing the national network.
2. **One hop at a time.** Direct trains, then one transfer, then two. Each step
   keeps the same output schema and adds one more temporal constraint.
3. **SFT then RLVR.** SFT copies the target format on the new prompt. RLVR
   (dense reward) then pushes `route`, `arrival_exact`, `chronology`, and
   `router`. Skip SFT and the policy often stays fluent but wrong.

### What each stage adds

| Stage | New skill | Context | Warm-start | Outcome |
|-------|-----------|---------|------------|---------|
| A | Select a real direct train from the prompt | Gold-assisted OD neighborhood | format SFT | `correct≈0.77` |
| B | Chain two verified legs + transfer wait | Gold-assisted 1-transfer | A RL | `correct≈0.71` |
| C | Chain three legs | Gold-assisted 2-transfer | B RL | `correct≈0.91` (best in curriculum) |
| D | Pick earliest arrival among valid candidates | 5–10 RAPTOR journeys | C RL | `optimal=0.997` (ranking only) |
| E | Same journey task, no gold retrieval | Deterministic expansion | C RL (not D) | `correct=0.85` |
| F | Explain the choice in `<think>` then answer | Same as E + thinking | E RL | `correct=0.92` (default) |

C is stronger than A/B even though it has more hops. That is expected under
curriculum transfer: C inherits A+B training and gets a more explicit
three-leg context. D is a side branch. It proves earliest-arrival comparison
is learnable, but its output is `Answer: Candidate E`, so it must **not**
warm-start E.

E is the production **retrieval** branch: same answer format as A/B/C, but the
prompt is a destination-constrained temporal subgraph (no RAPTOR facit).
RAPTOR stays verifier and chat fallback; it is not allowed to pick the context
trains. The default **checkpoint** is Experiment F RL (reasoning + journey) on
top of that retrieval.

F adds **visible reasoning** on top of E: the model is taught to write a short
Danish chain-of-thought before the journey. RLVR still grades only the final
journey text, not the prose quality of the thoughts.

### Training pattern (every stage)

```text
export JSONL  →  SFT on new prompts (1 epoch, LoRA)
              →  RLVR with dense reward (GRPO / importance sampling)
              →  held-out eval (unseen OD pairs)
```

SFT sequence length grows with the prompt: 4k for A/B/C/D, 32k for E
(~5.3k context tokens). RL only trains the assistant journey, not the
timetable block.

### What the curriculum does *not* cover

- **Alternative routes** (`alternativer` in chat) are a RAPTOR via-search, not
  a trained stage.
- **Delay / “what if”** questions are not in the curriculum.
- **3+ transfers** are mostly OOD: A–C train 0–2 skift; E/F inherit that
  distribution. Valid 3-skift answers can still fail `router` (wrong transfer
  count or slower than RAPTOR). Next curriculum step would mix 3-skift data and
  repair-style CoT (detect chronology error → replan).

## Prerequisites

1. **Tinker API key** — set `TINKER_API_KEY` (see [Tinker docs](https://tinker-docs.thinkingmachines.ai/)).
2. **Parsed timetable** — `all_services.jsonl` (from `dsb_timetable_parser.py`).
3. **Journey facit** — `journeys.jsonl` (from `generate_journeys.py`).
4. **Python env** — your `dsbpy` venv already has `tinker` and `tinker_cookbook`.

```powershell
cd C:\Users\I747069\Downloads\Tinker_Køreplan
.\dsbpy\Scripts\Activate.ps1
```

## Model: Qwen3.5-4B

Default model is **`Qwen/Qwen3.5-4B`** — well-tested in Tinker RL recipes (`math_rl`, `code_rl`).

- Renderer is auto-selected (`qwen3_5`).
- Journey prompts/answers are short for A/B/C (~80 / ~460 chars); use higher
  `max_tokens` (e.g. 3072) when thinking is on (F).
- Tinker Cookbook provides auto-LR formulas for Qwen3.5 if you want to tune further.

Override model on the CLI:

```powershell
python -m tinker_dsb_rl.train model_name="Qwen/Qwen3.5-4B"
```

## Step 0 — Export SFT splits (optional warm-start)

```powershell
python -m tinker_dsb_rl.export_sft_splits journeys.jsonl -o sft_data
```

Creates `sft_data/journeys_train.jsonl` (3372) and `sft_data/journeys_test.jsonl` (628).

## Step 1 — SFT warm-start (recommended)

Teach the output format before RL:

```powershell
python -m tinker_dsb_rl.train_sft `
  model_name="Qwen/Qwen3.5-4B" `
  file_path=sft_data/journeys_train.jsonl `
  learning_rate=5e-5 `
  batch_size=64 `
  lora_rank=32 `
  max_length=2048 `
  num_epochs=1 `
  wandb_project=DSB_timetable
```

Metrics log to WandB project **DSB_timetable** by default (override with `wandb_project=` or disable with `wandb_project=null`).

Use the saved checkpoint as `load_checkpoint_path` in the RL step.

## Step 2 — RLVR training

```powershell
python -m tinker_dsb_rl.train `
  model_name="Qwen/Qwen3.5-4B" `
  journeys_path=journeys.jsonl `
  services_path=all_services.jsonl `
  group_size=8 `
  groups_per_batch=32 `
  learning_rate=2e-5 `
  max_tokens=512 `
  kl_penalty_coef=0.05 `
  eval_every=20 `
  save_every=20 `
  wandb_project=DSB_timetable
```

Resume from SFT checkpoint:

```powershell
python -m tinker_dsb_rl.train `
  model_name="Qwen/Qwen3.5-4B" `
  load_checkpoint_path="tinker://YOUR_CHECKPOINT" `
  group_size=8 groups_per_batch=32 learning_rate=2e-5 max_tokens=512
```

### Key hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `group_size` | 8 | Completions per prompt (GRPO grouping). Try 4–16. |
| `groups_per_batch` | 32 | Prompts per optimizer step |
| `learning_rate` | 2e-5 | Lower if unstable |
| `kl_penalty_coef` | 0.05 | Keeps policy near base model |
| `max_tokens` | 512 | Journey answers need ~200–400 tokens |
| `use_router_reward` | true | Re-verify with RAPTOR router |

## RLVR reward breakdown

Each completion is scored in `journey_grading.py`:

| Component | Weight | Verifiable check |
|-----------|--------|------------------|
| Format | 0.10 | Numbered legs + Answer block |
| Arrival | 0.35 | Soft: 1.0 exact, linear → 0 at ±10 min |
| Transfers | 0.15 | `Transfers: N` matches facit |
| Route | 0.20 | Fraction of legs that exist as real timetable services (OD + chronology + `Timetable.verify_leg`) |
| Router | 0.20 | Matches `Timetable.earliest_journey()` |

Total reward ≈ 1.0 for a correct, verifiable journey. Invented but chronological routes score **0** on route (no free lunch).

Training uses **importance sampling** loss with group-relative advantages (standard Tinker RL loop).

## Held-out evaluation

Each experiment JSONL tags OD pairs by hash (15% held out as unseen origin–destination pairs):

| Dataset | Train | Test |
|---------|------:|-----:|
| `journeys_direct.jsonl` (A) | 8496 | 1504 |
| `journeys_transfer1.jsonl` (B) | 8376 | 1624 |
| `journeys_transfer2.jsonl` (C) | 8448 | 1552 |
| `journeys.jsonl` (legacy, no context) | 3374 | 628 |

The RL recipe runs eval on the test split every `eval_every` steps. Watch metrics
(also logged to WandB when `wandb_project` is set):

| WandB / metrics.jsonl key | Meaning |
|---------------------------|---------|
| `env/all/format` | Valid journey answer structure |
| `env/all/arrival` | Soft arrival score (1.0 exact → 0 at ±10 min) |
| `env/all/arrival_exact` | Binary exact arrival match |
| `env/all/arrival_err_min` | Absolute arrival error in minutes |
| `env/all/transfers` | Transfer-count match |
| `env/all/route` | Fraction of legs verified in the timetable |
| `env/all/router` | Matches RAPTOR earliest journey |
| `env/all/grade_total` | Weighted RLVR total (before format coef) |
| `env/all/correct` | `grade_total ≥ 0.85` |
| `env/all/reward` / `reward/total` | Training reward used by GRPO |

Same keys appear under `test/env/all/...` on eval steps.

### Offline eval CLI (`tinker_dsb_rl/eval.py`)

Re-runs the same `RLTestSetEvaluator` path as training (WandB-compatible metrics).
Uses the Exp C sampler checkpoint by default.

```powershell
# Smoke (16 held-out examples, Exp C only)
python -m tinker_dsb_rl.eval --limit 16

# All three experiments, 100 held-out each
python -m tinker_dsb_rl.eval --all-experiments --limit 100 --out eval_test100.json

# Full held-out eval (~4680 test rows total; slow)
python -m tinker_dsb_rl.eval --all-experiments --out eval_full.json
```

Default checkpoint (Experiment F RL — reasoning + journey):

```
tinker://f8535263-db47-523e-9434-05bb1742407b:train:0/sampler_weights/final
```

Previous default (Experiment E RL, no trained reasoning):

```
tinker://b0c0da2d-f455-585a-951f-09d2f8913c34:train:0/sampler_weights/final
```

Experiment C RL (warm-start for E):

```
tinker://4eb1e06b-c1f6-532b-b592-92a9984536a8:train:0/sampler_weights/final
```

#### Offline results — 100 held-out ODs per experiment (2026-08-30)

Checkpoint: Exp C RL final (sampler weights). Prompts use pre-baked timetable context
from each JSONL record (same as RL eval during training).

| Metric | A (0 xfer) | B (1 xfer) | C (2 xfer) |
|--------|------------|------------|------------|
| `route` | 0.91 | 0.94 | **0.95** |
| `arrival_exact` | 0.89 | 0.94 | **0.95** |
| `correct` | 0.87 | 0.88 | **0.91** |
| `router` | 0.71 | 0.68 | **0.80** |
| `grade_total` | 0.94 | 0.96 | **0.97** |
| `arrival_err_min` | ~1.5 min | ~0.3 min | **~0.4 min** |

**Read:** Exp C generalizes to unseen OD pairs — test `correct` (0.91) matches
training (~0.91). Main gap vs perfection is `router` on A/B (0.68–0.71): valid
routes with correct arrival, but not always RAPTOR-optimal.

Raw JSON: `eval_test100.json`

### Inference CLI (`tinker_dsb_rl/ask.py`)

Interactive pipeline: user query → retriever → model → RAPTOR verify.

**Default: Experiment E retrieval** (`retrieve_expanded_services`) and the
**Experiment F RL** checkpoint (thinking on by default). RAPTOR is used for
verification, `--show-facit`, and fallback, not for building the prompt context.
Use `--no-thinking` to disable reasoning traces; pass `--checkpoint` for E RL
if you need the previous silent policy.

```powershell
python -m tinker_dsb_rl.ask --from Humlebæk --to "Aarhus H" --after 08:00 --day 3
python -m tinker_dsb_rl.ask --from Humlebæk --to "Aarhus H" --after 08:00 --day 3 --show-facit
python -m tinker_dsb_rl.ask --from Humlebæk --to "Aarhus H" --after 08:00 --day 3 --json

# Training-style retrieval (uses RAPTOR facit to pick context trains)
python -m tinker_dsb_rl.ask --from Humlebæk --to "Aarhus H" --after 08:00 --day 3 --gold-retrieval

# Previous gold-free hub heuristic
python -m tinker_dsb_rl.ask --from Humlebæk --to "Aarhus H" --after 08:00 --day 3 --heuristic-retrieval
```

Gold-free retrieval strategy (`timetable_context.retrieve_query_services`):

1. Direct OD services in a time window
2. For top transfer hubs (reachable from origin + with trains to destination):
   origin→hub and hub→destination neighborhoods
3. For 2-transfer: a few hub pairs with three leg neighborhoods

Smoke tests (Humlebæk → Aarhus H, Wed 08:00):

| Retrieval | Result |
|-----------|--------|
| `--gold-retrieval` | Matches RAPTOR facit exactly (`correct=True`) |
| Default (gold-free) | Valid 1-transfer route, earlier train on leg 2 (`route=1.0`, `correct=False`) |

### Conversational CLI and fallback

```powershell
python -m tinker_dsb_rl.chat
```

The chat defaults to the Experiment E policy and expanded retrieval. After
every model response, the journey is checked against the timetable and RAPTOR
optimum. If verification returns `correct=False`, the model response is marked
as rejected and the verified RAPTOR journey is printed as the final fallback.

For debugging the raw model behavior, disable fallback explicitly:

```powershell
python -m tinker_dsb_rl.chat --no-fallback
```

## Local reward smoke test

```powershell
python -c "
from tinker_dsb_rl.journey_grading import parse_journey_response, grade_components
from dsb_routing import Timetable, load_trips, parse_hhmm

sample = open('journeys.jsonl', encoding='utf-8').readline()
import json; rec = json.loads(sample)
parsed = parse_journey_response(rec['messages'][1]['content'])
tt = Timetable(load_trips('all_services.jsonl'), weekday=rec['weekday'])
print(grade_components(
    parsed,
    origin=rec['origin'], destination=rec['destination'],
    reference_arrival=rec['arrival'], reference_transfers=rec['transfers'],
    depart_after=rec['depart_after'], weekday=rec['weekday'], timetable=tt,
))
"
```

## Three-stage recipe (SFT → RLVR → eval)

1. **SFT** on `journeys_train.jsonl` — learn Danish journey answer format.
2. **RLVR** on train split — optimize verifiable routing reward with GRPO.
3. **Eval** on test split — measure generalization to unseen OD pairs (e.g. Humlebæk→Skagen held out).

This matches the original plan: the model learns temporal graph reasoning, not memorization.

## Training log (Qwen3.5-4B)

### SFT — 3 epochs (`sft-qwen-3ep`)

```powershell
python -m tinker_dsb_rl.train_sft file_path=sft_data/journeys_train.jsonl num_epochs=3 wandb_project=DSB-journey wandb_name=sft-qwen-3ep behavior_if_log_dir_exists=delete
```

- **Checkpoint:** `tinker://bcba9f5d-5a32-50d6-a15c-0c7e915d258a:train:0/weights/final`
- **WandB:** https://wandb.ai/sojohan/DSB-journey/runs/ijnv8aqi
- Train/test BPB ~0.14 (vs ~0.23 after 1 epoch). Format is solid.
- Held-out sampling still scored **route 0 / arrival 0** — fluent style, invented trains.

### RLVR — from 3-epoch SFT (`rl-from-sft-3ep`)

Warm-start: `load_checkpoint_path="tinker://bcba9f5d-5a32-50d6-a15c-0c7e915d258a:train:0/weights/final"`

- **Final checkpoint:** `tinker://f4df910b-16c7-5198-b80c-4705b09251c1:train:0/weights/final`
- **WandB:** https://wandb.ai/sojohan/DSB-journey/runs/9tfwaka5
- Training completed at step 105.

#### Final scores (step 105)

| Metric | Value | Read |
|--------|-------|------|
| `format` | **1.00** | Always valid structure |
| `transfers` | **0.33** | Transfer count right ~1/3 of the time |
| `route` | **0.00** | No legs verified in the timetable |
| `arrival` (soft) | **0.007** | Essentially never near facit |
| `arrival_exact` | **0** | Never exact |
| `arrival_err_min` | **~148 min** | ~2.5 h wrong on average |
| `router` / `correct` | **0** | Never matches RAPTOR / never ≥0.85 |
| `reward` / `grade_total` | **0.15** | ≈ format (0.10) + some transfers |

`frac_all_good = 0` for the whole run — no group ever had all-good completions. `frac_mixed = 0.83` means GRPO mostly saw similarly mediocre samples, so the routing advantage signal was weak.

#### vs previous RL (hard route check)

| | Earlier RL | `rl-from-sft-3ep` |
|--|------------|-------------------|
| Warm-start | 1-epoch SFT | **3-epoch SFT** |
| `reward` | ~0.13 | **0.15** |
| `route` | 0 | **0** |
| `transfers` | ~0.19 | **0.33** |
| `arrival_err_min` | ~181 | **~148** |
| `correct` | 0 | **0** |

Slightly better transfer guessing and a bit less arrival error, but **no breakthrough**. Extra SFT epochs did not unlock timetable accuracy under RL.

#### Sparkline takeaway

- `correct` flat at 0 the entire run
- `format` already near-perfect early
- `arrival` / `grade_total` bounce without a clear upward trend
- KL vs base stayed small (`~0.013`) — policy didn’t drift far; it also didn’t discover better routes

### Verdict

RLVR from this setup is **stuck**: the model can emit journey-shaped text, but almost never samples a real service chain, so `route`/`arrival`/`router` stay near zero and there is nothing useful for GRPO to reinforce.

**Do not re-run the same recipe.** See **Curriculum training** above, then
**Experiment A** (retrieval + 0-transfer + dense reward).

## Experiment A — retrieval + 0-transfer + dense reward

Goal: prove the model can **select** a real direct train when the relevant timetable is in the prompt.

| Piece | Choice |
|-------|--------|
| Task | **0 transfers only** |
| Prompt | Local timetable services (≤30 trains) + question |
| Data | `journeys_direct.jsonl` (~10k) |
| Reward | `reward_mode=dense` (partial credit) |
| Success bar | `route > 0.80`, `arrival_exact > 0.60`, `correct > 0.50` |

### Generate data

```powershell
python generate_journeys.py generate-direct all_services.jsonl `
  -o journeys_direct.jsonl --count 10000 --max-context 30

python -m tinker_dsb_rl.export_sft_splits journeys_direct.jsonl -o sft_data_direct
```

Current build: **10000** journeys (8496 train / 1504 test), mean **~11** context services per prompt, all `transfers=0`.

### Dense reward weights

| Component | Weight |
|-----------|--------|
| Format | 0.10 |
| Valid stations | 0.05 |
| Valid train / verified leg label | 0.10 |
| First leg verified in timetable | 0.20 |
| Chronology + OD endpoints | 0.10 |
| Transfers match | 0.10 |
| Correct destination | 0.10 |
| Soft arrival | 0.15 |
| Optimal / router | 0.10 |

### Train (SFT → RLVR)

```powershell
# SFT warm-start on context prompts
python -m tinker_dsb_rl.train_sft `
  file_path=sft_data_direct/journeys_train.jsonl `
  num_epochs=1 `
  max_length=4096 `
  wandb_project=DSB-journey `
  wandb_name=sft-expA-direct `
  behavior_if_log_dir_exists=delete

# RLVR with dense reward
python -m tinker_dsb_rl.train `
  journeys_path=journeys_direct.jsonl `
  services_path=all_services.jsonl `
  reward_mode=dense `
  max_tokens=1024 `
  load_checkpoint_path="tinker://YOUR_SFT_CHECKPOINT" `
  wandb_project=DSB-journey `
  wandb_name=rl-expA-direct `
  behavior_if_log_dir_exists=delete
```

Extra metrics when `reward_mode=dense`: `env/all/leg0`, `valid_station`, `valid_train`, `chronology`, `dest_ok`.

### Results — SFT (`sft-expA-direct`)

- **Checkpoint:** `tinker://bf6a1b6b-b150-54cc-88f0-b7394af6b7be:train:0/weights/final`
- **WandB:** https://wandb.ai/sojohan/DSB-journey/runs/otiop6na
- 1 epoch, 128 steps. Final `train_mean_bpb` **0.0038**, `test/bpb` **0.0052** (train ≈ test; no obvious overfitting).

### Results — RLVR (`rl-expA-direct`)

Warm-start: `tinker://bf6a1b6b-b150-54cc-88f0-b7394af6b7be:train:0/weights/final`

- **Final checkpoint:** `tinker://c13b7fe0-c1f5-586a-b14a-6616d9c8ec65:train:0/weights/final`
- **WandB:** https://wandb.ai/sojohan/DSB-journey/runs/agtxmcsp
- Completed at step 265.

#### Success bar (step 265)

| Target | Result | |
|--------|--------|--|
| `route > 0.80` | **0.94** | Pass |
| `arrival_exact > 0.60` | **0.77** | Pass |
| `correct > 0.50` | **0.77** | Pass |

#### Final scores

| Metric | Value | Read |
|--------|-------|------|
| `format` | **1.00** | Always structured |
| `route` / `leg0` | **0.94** | Real verified direct legs |
| `arrival` / `arrival_exact` | **0.77** | Exact arrival most of the time |
| `router` | **0.70** | Often matches RAPTOR earliest |
| `reward` / `grade_total` | **0.93** | Dense signal working |
| `frac_all_good` | **0.94** | GRPO groups mostly all-good |
| `frac_all_bad` | **0** | No dead groups |
| `arrival_err_min` | ~18 | When wrong, errors still sizable |

#### Verdict

**Experiment A passed.** Vs old no-context RL (`route=0`, `correct=0`, `reward≈0.15`): the model is selecting from the prompt, not inventing trains. Entropy ended very low (`~0.007`).

## Experiment B — 1 transfer + dual-leg context

Goal: select a **two-leg** journey (origin → hub → destination) when both leg neighborhoods are in the prompt.

| Piece | Choice |
|-------|--------|
| Task | **Exactly 1 transfer** |
| Prompt | ≤40 services = origin→hub candidates + hub→dest candidates |
| Data | `journeys_transfer1.jsonl` (~10k) |
| Reward | `reward_mode=dense` (includes `leg0` + `leg1`) |
| Warm-start | Experiment A RL checkpoint (recommended) |
| Success bar | `route > 0.70`, `arrival_exact > 0.50`, `correct > 0.40` |

### Generate data

```powershell
python generate_journeys.py generate-transfer1 all_services.jsonl `
  -o journeys_transfer1.jsonl --count 10000 --max-context 40

python -m tinker_dsb_rl.export_sft_splits journeys_transfer1.jsonl -o sft_data_transfer1
```

Current build: **10000** journeys (8376 train / 1624 test), mean **~22** context services, all `transfers=1`. Top hubs: København H, Aarhus H, Odense, Fredericia, …

### Dense reward (A/B)

| Component | Weight |
|-----------|--------|
| Valid stations | 0.05 |
| Valid train / verified label | 0.10 |
| First leg verified (`leg0`) | 0.15 |
| Second leg verified (`leg1`) | 0.15 |
| Chronology + min skiftetid | 0.10 |
| Transfers match | 0.10 |
| Correct destination | 0.10 |
| Soft arrival | 0.15 |
| Optimal / router | 0.10 |

On direct (A) journeys, `leg1` mirrors `leg0` so the scale stays comparable.

### Train (SFT → RLVR)

```powershell
# SFT on 1-transfer context prompts (optional warm-start from Exp A)
python -m tinker_dsb_rl.train_sft `
  file_path=sft_data_transfer1/journeys_train.jsonl `
  load_checkpoint_path="tinker://c13b7fe0-c1f5-586a-b14a-6616d9c8ec65:train:0/weights/final" `
  num_epochs=1 `
  max_length=4096 `
  wandb_project=DSB-journey `
  wandb_name=sft-expB-transfer1 `
  behavior_if_log_dir_exists=delete

# RLVR
python -m tinker_dsb_rl.train `
  journeys_path=journeys_transfer1.jsonl `
  services_path=all_services.jsonl `
  reward_mode=dense `
  max_tokens=1024 `
  load_checkpoint_path="tinker://e5ef1533-4df6-5af6-850b-9304363c994d:train:0/weights/final" `
  wandb_project=DSB-journey `
  wandb_name=rl-expB-transfer1 `
  behavior_if_log_dir_exists=delete
```

Or skip SFT and RL directly from the A checkpoint:

```powershell
python -m tinker_dsb_rl.train `
  journeys_path=journeys_transfer1.jsonl `
  services_path=all_services.jsonl `
  reward_mode=dense `
  max_tokens=1024 `
  load_checkpoint_path="tinker://c13b7fe0-c1f5-586a-b14a-6616d9c8ec65:train:0/weights/final" `
  wandb_project=DSB-journey `
  wandb_name=rl-expB-fromA `
  behavior_if_log_dir_exists=delete
```

Watch `env/all/leg0`, `env/all/leg1`, `route`, `arrival_exact`, `correct`.

### Results — SFT (`sft-expB-transfer1`)

Warm-start: Exp A RL `tinker://c13b7fe0-c1f5-586a-b14a-6616d9c8ec65:train:0/weights/final`

- **Checkpoint:** `tinker://e5ef1533-4df6-5af6-850b-9304363c994d:train:0/weights/final`
- **WandB:** https://wandb.ai/sojohan/DSB-journey/runs/ypiauzkg
- 1 epoch, 126 steps. Final `train_mean_bpb` **0.013**, `test/bpb` **0.009** (slightly higher than A — longer 1-transfer answers; train ≈ test).

### Results — RLVR (`rl-expB-transfer1`)

Warm-start: `tinker://e5ef1533-4df6-5af6-850b-9304363c994d:train:0/weights/final`

- **Final checkpoint:** `tinker://cc5fa17c-0795-516c-9a7e-a2d5940b85fd:train:0/weights/final`
- **WandB:** https://wandb.ai/sojohan/DSB-journey/runs/orakvfu5
- Completed at step 261.

#### Success bar (step 261)

| Target | Result | |
|--------|--------|--|
| `route > 0.70` | **0.88** | Pass |
| `arrival_exact > 0.50` | **0.76** | Pass |
| `correct > 0.40` | **0.71** | Pass |

#### Final scores

| Metric | Value | Read |
|--------|-------|------|
| `format` | **1.00** | Always structured |
| `leg0` / `leg1` | **0.88 / 0.89** | Both legs usually real |
| `route` | **0.88** | Verified multi-leg chains |
| `arrival` / `arrival_exact` | 0.88 / **0.76** | Soft strong; exact often right |
| `arrival_err_min` | **~1.4** | When wrong, only slightly off |
| `router` | **0.55** | Optimal RAPTOR ~half the time |
| `reward` / `grade_total` | **0.91** | Dense signal healthy |
| `transfers` / `dest_ok` | **1.00** | Always 1 transfer + right dest |
| `frac_all_good` | **0.58** | More mixed than A (harder) |
| `frac_all_bad` | **0** | No dead groups |

#### vs Experiment A

| | A (direct) | B (1 transfer) |
|--|------------|----------------|
| `route` | 0.94 | **0.88** |
| `arrival_exact` | 0.77 | **0.76** |
| `correct` | 0.77 | **0.71** |
| `router` | 0.70 | **0.55** |
| `arrival_err_min` | ~18 | **~1.4** |

#### Verdict

**Experiment B passed.** Slightly harder than A (expected), but transfer reasoning works. Remaining gap is mostly **optimal** choice (`router`), not illegal routes.

## Experiment C — 2 transfers + three-leg context

Goal: select a **three-leg** journey (origin → hub₁ → hub₂ → destination) when all leg neighborhoods are in the prompt.

| Piece | Choice |
|-------|--------|
| Task | **Exactly 2 transfers** |
| Prompt | ≤50 services across the three legs |
| Data | `journeys_transfer2.jsonl` (~10k) |
| Reward | `reward_mode=dense` (`leg0` + `leg1` + `leg2`) |
| Warm-start | Experiment B RL checkpoint (recommended) |
| Success bar | `route > 0.60`, `arrival_exact > 0.40`, `correct > 0.30` |

### Generate data

```powershell
python generate_journeys.py generate-transfer2 all_services.jsonl `
  -o journeys_transfer2.jsonl --count 10000 --max-context 50

python -m tinker_dsb_rl.export_sft_splits journeys_transfer2.jsonl -o sft_data_transfer2
```

Current build: **10000** journeys (8448 train / 1552 test), mean **~31** context services, all `transfers=2`. Top hubs: København H, Aarhus H, Fredericia, Odense, …

### Dense reward (A/B/C)

| Component | Weight |
|-----------|--------|
| Valid stations | 0.05 |
| Valid train / verified label | 0.07 |
| First leg verified (`leg0`) | 0.12 |
| Second leg verified (`leg1`) | 0.12 |
| Third leg verified (`leg2`) | 0.12 |
| Chronology + min skiftetid | 0.10 |
| Transfers match | 0.08 |
| Correct destination | 0.09 |
| Soft arrival | 0.15 |
| Optimal / router | 0.10 |

Unused leg slots are mirrored (A) or averaged (B) so older experiments stay on a comparable scale.

### Train (SFT → RLVR)

```powershell
# SFT from Exp B RL checkpoint
python -m tinker_dsb_rl.train_sft `
  file_path=sft_data_transfer2/journeys_train.jsonl `
  load_checkpoint_path="tinker://cc5fa17c-0795-516c-9a7e-a2d5940b85fd:train:0/weights/final" `
  num_epochs=1 `
  max_length=4096 `
  wandb_project=DSB-journey `
  wandb_name=sft-expC-transfer2 `
  behavior_if_log_dir_exists=delete

# RLVR
python -m tinker_dsb_rl.train `
  journeys_path=journeys_transfer2.jsonl `
  services_path=all_services.jsonl `
  reward_mode=dense `
  max_tokens=1536 `
  load_checkpoint_path="tinker://1d62165a-071c-5a0f-926c-843dc852449e:train:0/weights/final" `
  wandb_project=DSB-journey `
  wandb_name=rl-expC-transfer2 `
  behavior_if_log_dir_exists=delete
```

Watch `env/all/leg0`, `leg1`, `leg2`, `route`, `arrival_exact`, `correct`.

### Results — SFT (`sft-expC-transfer2`)

Warm-start: Exp B RL `tinker://cc5fa17c-0795-516c-9a7e-a2d5940b85fd:train:0/weights/final`

- **Checkpoint:** `tinker://1d62165a-071c-5a0f-926c-843dc852449e:train:0/weights/final`
- **WandB:** https://wandb.ai/sojohan/DSB-journey/runs/bp3yi3cu
- 1 epoch. Final `test/bpb` **0.0047** (~195k tokens/batch — longer 3-leg prompts).

### Results — RLVR (`rl-expC-transfer2`)

Warm-start: `tinker://1d62165a-071c-5a0f-926c-843dc852449e:train:0/weights/final`

- **Final checkpoint:** `tinker://4eb1e06b-c1f6-532b-b592-92a9984536a8:train:0/weights/final`
- **WandB:** https://wandb.ai/sojohan/DSB-journey/runs/hqmlus0v
- Completed at step 263.

#### Success bar (step 263)

| Target | Result | |
|--------|--------|--|
| `route > 0.60` | **0.93** | Pass |
| `arrival_exact > 0.40` | **0.89** | Pass |
| `correct > 0.30` | **0.91** | Pass |

#### Final scores

| Metric | Value | Read |
|--------|-------|------|
| `format` | **1.00** | Always structured |
| `leg0` / `leg1` / `leg2` | **0.94 / 0.93 / 0.92** | All three legs usually real |
| `route` | **0.93** | Verified 3-leg chains |
| `arrival` / `arrival_exact` | 0.98 / **0.89** | Very strong timing |
| `arrival_err_min` | **~0.2** | When wrong, almost negligible (minutes) |
| `router` | **0.77** | Often matches RAPTOR optimum |
| `reward` / `grade_total` | **0.96** | Dense signal healthy |
| `transfers` / `dest_ok` | **1.00** | Always 2 transfers + right dest |
| `frac_all_good` | **0.75** | 3/4 of GRPO groups all-good |
| `frac_all_bad` | **0** | No dead groups |

#### vs Experiments A and B

| | A (0 xfer) | B (1 xfer) | C (2 xfer) |
|--|------------|------------|------------|
| `route` | 0.94 | 0.88 | **0.93** |
| `arrival_exact` | 0.77 | 0.76 | **0.89** |
| `correct` | 0.77 | 0.71 | **0.91** |
| `router` | 0.70 | 0.55 | **0.77** |
| `arrival_err_min` | ~18 | ~1.4 | **~0.2** |

#### Verdict

**Experiment C passed strongly** — best run in the curriculum. Three-leg chaining works reliably with retrieval + dense RLVR. Offline held-out eval (100 unseen ODs per experiment) confirms Exp C generalizes (`correct` 0.91 on test, matching training).

**Curriculum complete (A → B → C).** Production use: retrieve local services → prompt → model → optional RAPTOR verify (`tinker_dsb_rl/ask.py`).

## Experiment D — rank valid candidate journeys

Goal: isolate **optimality** from retrieval and path construction. Each prompt
contains 5–10 valid candidate journeys in shuffled order. The model only selects
the candidate with the earliest arrival:

```text
Answer: Candidate E
```

### Reward

| Component | Weight | Meaning |
|-----------|-------:|---------|
| `valid_candidate` | 0.20 | Output names one candidate present in the prompt |
| `optimal` | 0.80 | Selected candidate has the minimum arrival time |

`arrival_regret_min` measures how many minutes later the selected candidate
arrives than the optimum. A separate format coefficient penalizes malformed
outputs.

### Generate data

The generator reuses queries from A/B/C and derives distinct valid alternatives
by advancing through real departure events at the origin.

```powershell
# Small setup check
python generate_ranking.py all_services.jsonl `
  -o journeys_ranking_smoke.jsonl `
  --stats journey_ranking_smoke_stats.json `
  -n 100

# Full dataset
python generate_ranking.py all_services.jsonl `
  -o journeys_ranking.jsonl `
  --stats journey_ranking_stats.json `
  -n 10000

python -m tinker_dsb_rl.export_sft_splits `
  journeys_ranking.jsonl -o sft_data_ranking
```

### Train (Exp C → D SFT → D RLVR)

```powershell
# Teach the new short ranking output format.
python -m tinker_dsb_rl.train_sft `
  file_path=sft_data_ranking/journeys_train.jsonl `
  load_checkpoint_path="tinker://4eb1e06b-c1f6-532b-b592-92a9984536a8:train:0/weights/final" `
  max_length=4096 `
  num_epochs=1 `
  wandb_project=DSB-journey `
  wandb_name=sft-expD-ranking `
  behavior_if_log_dir_exists=delete

# Replace <D-SFT-CHECKPOINT> with the checkpoint printed above.
python -m tinker_dsb_rl.train_ranking `
  rankings_path=journeys_ranking.jsonl `
  load_checkpoint_path="<D-SFT-CHECKPOINT>" `
  wandb_project=DSB-journey `
  wandb_name=rl-expD-ranking `
  behavior_if_log_dir_exists=delete
```

Watch `env/all/valid_candidate`, `optimal`, `arrival_regret_min`, and `correct`;
held-out versions are logged under `test/env/all/...`.

### Baseline smoke result (before D training)

Exp C final checkpoint on the 14 held-out examples in the 100-row smoke dataset:

| Metric | Value |
|--------|------:|
| `format` | 1.000 |
| `valid_candidate` | 1.000 |
| `optimal` / `correct` | **0.929** |
| `arrival_regret_min` | 4.29 min |

Raw result: `eval_ranking_smoke.json`.

The full dataset contains 10,000 examples (8,489 train / 1,511 held-out test)
with 5–10 candidates per prompt (mean 9.44).

### Results — ranking SFT

Warm-start: Exp C RL final.

- **Checkpoint:** `tinker://8171a3f5-f89b-5c28-a1e5-2a8d1d765e9a:train:0/weights/final`
- **Sampler:** `tinker://8171a3f5-f89b-5c28-a1e5-2a8d1d765e9a:train:0/sampler_weights/final`
- **WandB:** https://wandb.ai/sojohan/DSB-journey/runs/6yk0k9ct
- One SFT epoch; final `test/bpb` 0.00013.

#### Held-out evaluation (all 1,511 test examples)

| Metric | Exp C baseline (100) | D SFT (1,511) |
|--------|---------------------:|--------------:|
| `format` | 0.98 | **1.0000** |
| `valid_candidate` | 0.98 | **1.0000** |
| `optimal` / `correct` | 0.76 | **0.9967** |
| `arrival_regret_min` | 36.63 min | **0.86 min** |
| `reward` | 0.803 | **0.9974** |

Raw full result: `eval_ranking_sft_full.json`.

Only 5 of 1,511 held-out examples were incorrect. However, those rare misses
are relatively costly: the overall 0.86-minute mean regret implies about 260
minutes regret per failed example. Inspect those five cases before using the
ranker without a deterministic fallback.

#### Verdict

**Experiment D passed.** Explicit earliest-arrival comparison is learnable
separately from route construction. Ranking-specific SFT raised held-out
optimal selection from 0.76 to 0.9967.

Do not run D RLVR yet: the SFT policy is already saturated
(`frac_all_good=0.9967`), so group-relative RL would provide almost no useful
advantage signal. This experiment isolates candidate ranking; it does not by
itself prove that gold-free retrieval and candidate construction are solved.

## Experiment E — deterministic temporal graph expansion

Goal: replace the heuristic gold-free retriever with an exhaustive, bounded
local timetable graph:

```text
origin + destination + departure time
  -> deterministic temporal expansion
  -> destination-constrained service graph
  -> LLM journey
  -> RAPTOR verification
```

The retriever does **not** inspect RAPTOR's journey, rank hubs, or optimize
arrival time. It first computes a static reverse-reachability closure from the
destination. Forward expansion then keeps every boardable service that can
still reach the destination within the remaining leg count. Fixed bounds limit
the search to:

- at most 2 transfers / 3 vehicle legs;
- departures within 240 minutes of each reached state;
- a 720-minute total time horizon;
- the earliest 3 temporal states per station;
- at most 200 context services, balanced across hop depth and board station.

### Coverage gate before training

```powershell
python -m tinker_dsb_rl.eval_expansion `
  --limit 100 `
  --out eval_expansion_coverage100.json
```

Held-out smoke result after fixing RAPTOR path reconstruction:

| Metric | Heuristic gold-free | Experiment E |
|--------|--------------------:|-------------:|
| Gold journey fully in context | 0.49 | **1.00** |
| 0-transfer coverage | — | **1.00** |
| 1-transfer coverage | — | **1.00** |
| 2-transfer coverage | — | **1.00** |
| Mean context services | 50 max | 132.3 |
| Rough mean context tokens | — | 5,321 |

Raw result: `eval_expansion_coverage100.json`.

This is a strong smoke result, but not yet a trained Experiment E result. Run
coverage on the full held-out set before training. Uncovered examples should
not enter SFT because their target journey is absent from the prompt.

### Generate and use E data

```powershell
python generate_expansion.py all_services.jsonl `
  -o journeys_expansion.jsonl `
  --stats journey_expansion_stats.json `
  -n 10000

python -m tinker_dsb_rl.export_sft_splits `
  journeys_expansion.jsonl -o sft_data_expansion
```

The generator excludes uncovered examples by default and records context
coverage, truncation, service count, and reachable-state count per example.

### Results — E SFT

Warm-start: Exp C RL final. 32k sequence length, batch 4.

| Run | Train examples | Checkpoint (sampler) | WandB |
|-----|---------------:|----------------------|-------|
| 1k pilot | 827 | `tinker://4246f3ff-e225-5146-a662-32faff3de5e9:train:0/sampler_weights/final` | [v3d6262h](https://wandb.ai/sojohan/DSB-journey/runs/v3d6262h) |
| 5k | 4,210 | `tinker://154bb3b9-1fd9-5f22-9ddc-289b84353cc2:train:0/sampler_weights/final` | [1ckbx11c](https://wandb.ai/sojohan/DSB-journey/runs/1ckbx11c) |
| 5k + RLVR | 4,210 | `tinker://b0c0da2d-f455-585a-951f-09d2f8913c34:train:0/sampler_weights/final` | [me273z9f](https://wandb.ai/sojohan/DSB-journey/runs/me273z9f) |

#### Held-out evaluation (100 test examples)

| Metric | C baseline | E SFT 1k | E SFT 5k | E RL |
|--------|----------:|---------:|---------:|-----:|
| `format` | 1.000 | 1.000 | 1.000 | **1.000** |
| `route` | 0.198 | 0.721 | 0.882 | **0.930** |
| `arrival_exact` | 0.500 | 0.790 | 0.840 | **0.890** |
| `chronology` | 0.250 | 0.850 | 0.970 | **0.980** |
| `router` | 0.190 | 0.620 | 0.720 | **0.820** |
| `correct` | 0.150 | 0.540 | 0.730 | **0.850** |
| `reward` | 0.593 | 0.835 | 0.909 | **0.949** |

Raw results: `eval_expansion_baseline100.json`, `eval_expansion_sft100.json`,
`eval_expansion_sft5k_100.json`, `eval_expansion_rl100.json`.

#### Verdict

**Experiment E passed.** A destination-constrained temporal expansion, with no
RAPTOR facit in the prompt, is enough for the model to learn routing. RLVR
raised held-out `correct` from 0.73 to 0.85 and `router` from 0.72 to 0.82.

The remaining gap is optimality, not format or chronology. Keep RAPTOR as
verifier/fallback in chat. The **F RL** sampler is now the default checkpoint
(thinking on). E RL remains available via `--checkpoint`.

```powershell
python -m tinker_dsb_rl.chat
python -m tinker_dsb_rl.ask --from Humlebæk --to Ikast --after 08:00
```

### Alternative routes in chat

`alternativer` is a separate RAPTOR tool, not an Experiment E model skill. The
chat detects questions such as *Hvilke alternative ruter er der fra X til Y?*
or the follow-up `alternativer`. It then:

1. takes the unconstrained RAPTOR optimum;
2. forces extra journeys through major hubs that have trains to the destination
   (Skanderborg, Herning, Aarhus H, …);
3. keeps the earliest journey per last-board hub so the list is
   geographically diverse, not later departures of the same path.

Worked example, Humlebæk → Ikast after 08:00 (Wednesday):

| Choice | Via | Arrival | Role |
|--------|-----|---------|------|
| A | Herning | 13:12 | RAPTOR earliest; E model also selected this |
| B | Skanderborg | 13:25 | Distinct east-side path |
| C | Aarhus H | 14:04 | Longer detour |

A and B are the useful pair for presentations. C–E are valid but slower.
State clearly that the list is RAPTOR-composed; the LLM only produced the
first single journey.

```powershell
# After a planned trip:
alternativer

# Or in one question:
Hvilke alternative ruter er der fra Humlebæk til Ikast?
```

## Experiment F — reasoning traces (think, then answer)

Goal: make the product feel like a **reasoning planner**, not only a silent
selector over a timetable snippet. The model should emit a short Danish
`<think>` block (transfer times, why this train) and then the same journey
format as Experiment E.

| Piece | Choice |
|-------|--------|
| Warm-start | Experiment E RL sampler weights |
| Context | Same expanded retrieval as E |
| New skill | Visible chain-of-thought before `Answer:` |
| Reward (RL later) | Dense journey metrics only — not “nice prose” |

### Step 1 — export reasoning SFT data

Synthetic Danish reasoning is built from RAPTOR facit legs (deterministic,
no teacher API). Assistant content is a **plain string** (HF datasets cannot
mix list and string `content` fields):

```text
<think>
Etappe 1: …
Skift i X: …
</think>

<format_assistant journey>
```

```powershell
python -m tinker_dsb_rl.export_reasoning_sft `
  journeys_expansion_5k.jsonl `
  -o sft_data_reasoning
```

Smoke (20 rows):

```powershell
python -m tinker_dsb_rl.export_reasoning_sft `
  journeys_expansion_5k.jsonl `
  -o sft_data_reasoning_smoke `
  --limit 20
```

### Step 2 — SFT warm-start from E

Use renderer `qwen3_5` so thinking parts render as `<think>…</think>`:

```powershell
python -m tinker_dsb_rl.train_sft `
  file_path=sft_data_reasoning/journeys_train.jsonl `
  load_checkpoint_path="tinker://b0c0da2d-f455-585a-951f-09d2f8913c34:train:0/weights/final" `
  renderer_name=qwen3_5 `
  max_length=32768 `
  batch_size=4 `
  num_epochs=1 `
  wandb_project=DSB-journey `
  wandb_name=sft-expF-reasoning `
  behavior_if_log_dir_exists=delete
```

#### F SFT result (2026-09-08)

| | |
|--|--|
| WandB | https://wandb.ai/sojohan/DSB-journey/runs/xiosxt8l |
| Weights | `tinker://931bcb37-5a47-588d-8e92-517307957bcf:train:0/weights/final` |
| Sampler | `tinker://931bcb37-5a47-588d-8e92-517307957bcf:train:0/sampler_weights/final` |
| `test/nll` | ~0.0014 |
| `train_mean_nll` (final) | ~0.0017 |

SFT completed successfully. Held-out eval + RLVR: see Step 4.

#### F SFT held-out eval (100, thinking on)

Source: `eval_expansion_f_sft100.json`

| Metric | F SFT | E RL (ref) |
|--------|------:|-----------:|
| `format` | 1.00 | 1.00 |
| `route` | 0.93 | 0.93 |
| `arrival_exact` | 0.91 | 0.89 |
| `chronology` | 0.97 | 0.98 |
| `router` | **0.85** | 0.82 |
| `correct` | **0.86** | 0.85 |
| `reward` | 0.95 | 0.95 |

Reasoning SFT did **not** hurt journey quality vs E — slightly better
`correct`/`router` with thinking enabled. Remaining gap is optimality; RLVR
can still help if rollouts stay diverse (`temperature=1`).

### Step 3 — try reasoning in chat / UI

Thinking is on by default after F RL became production. Optional flags:

```powershell
python -m tinker_dsb_rl.chat
# or explicitly:
python -m tinker_dsb_rl.chat --thinking --show-reasoning

python -m tinker_dsb_rl.ask --from Humlebæk --to Skive --after 08:00
# disable: --no-thinking
```

Web UI: `POST /api/plan` with `thinking=true` (default). The LLM tab shows a
collapsible **Modellens reasoning** block. See **Web UI** below for how to start
the servers.

### Step 4 — held-out eval + RLVR on F

F SFT held-out eval (same E test split; thinking on; grades answer text only):

```powershell
python -m tinker_dsb_rl.eval `
  --experiment F `
  --journeys journeys_expansion.jsonl `
  --checkpoint "tinker://931bcb37-5a47-588d-8e92-517307957bcf:train:0/sampler_weights/final" `
  --renderer qwen3_5 `
  --thinking `
  --max-tokens 3072 `
  --limit 100 `
  --out eval_expansion_f_sft100.json
```

F RL held-out eval (production default):

```powershell
python -X utf8 -m tinker_dsb_rl.eval `
  --experiment F `
  --journeys journeys_expansion.jsonl `
  --checkpoint "tinker://f8535263-db47-523e-9434-05bb1742407b:train:0/sampler_weights/final" `
  --renderer qwen3_5 `
  --thinking `
  --max-tokens 3072 `
  --limit 100 `
  --out eval_expansion_f_rl100.json
```

RLVR warm-start from F SFT **weights** (not sampler). Keep grading on the
journey answer only (`get_text_content` strips `<think>`). Do **not** reward
thinking length/style:

```powershell
python -X utf8 -m tinker_dsb_rl.train `
  journeys_path=journeys_expansion_5k.jsonl `
  services_path=all_services.jsonl `
  reward_mode=dense `
  enable_thinking=true `
  renderer_name=qwen3_5 `
  max_tokens=3072 `
  group_size=4 `
  groups_per_batch=8 `
  load_checkpoint_path="tinker://931bcb37-5a47-588d-8e92-517307957bcf:train:0/weights/final" `
  wandb_project=DSB-journey `
  wandb_name=rl-expF-reasoning `
  behavior_if_log_dir_exists=delete
```

#### F RL result (2026-09-08)

Completed **527** batches (terminal log truncated at batch 5 — run was not stuck).

| | |
|--|--|
| Weights | `tinker://f8535263-db47-523e-9434-05bb1742407b:train:0/weights/final` |
| Sampler | `tinker://f8535263-db47-523e-9434-05bb1742407b:train:0/sampler_weights/final` |
| Periodic | `…/sampler_weights/000520` (and every 20) |

Held-out test during RL (batch 520):

| Metric | F SFT eval | F RL (test@520) |
|--------|----------:|----------------:|
| `correct` | 0.86 | **0.91** |
| `router` | 0.85 | **0.86** |
| `route` | 0.93 | 0.95 |
| `arrival_exact` | 0.91 | 0.94 |
| `reward` | 0.95 | 0.97 |

#### F RL clean held-out eval (100)

Source: `eval_expansion_f_rl100.json`

| Metric | F SFT | F RL (clean) | E RL (ref) |
|--------|------:|-------------:|-----------:|
| `format` | 1.00 | **1.00** | 1.00 |
| `route` | 0.93 | **0.95** | 0.93 |
| `arrival_exact` | 0.91 | **0.94** | 0.89 |
| `chronology` | 0.97 | **0.99** | 0.98 |
| `router` | 0.85 | **0.88** | 0.82 |
| `correct` | 0.86 | **0.92** | 0.85 |
| `reward` | 0.95 | **0.97** | 0.95 |

Clean held-out confirms the mid-run test@520 signal: F RL beats both F SFT
and E RL on `correct` / `router` with thinking enabled.

**Default checkpoint** is now F RL (`DEFAULT_SAMPLER` in `inference.py`);
thinking + show-reasoning default on in `ask`/`chat`/UI.

#### OOD smoke (manual UI)

Curriculum is strongest on 0–2 transfers. Quick checks beyond that:

| Query | Observation |
|-------|-------------|
| Esbjerg → Roskilde after 08:00 | In-distribution; model matched RAPTOR (1 skift) |
| Alken → Næstved after 12:00 | 3 skift, same arrival as RAPTOR → `godkendt=ja` |
| Frederikshavn → Esbjerg after 12:00 | Valid 2-skift path; 3-skift RAPTOR is ~28 min earlier |
| Langå → Tønder after 09:15 | Same arrival, one extra skift vs RAPTOR |
| Roskilde → Grenaa after 09:15 | CoT spotted bad chronology but still used it → `rute=0` |
| Asnæs → Skanderborg after 16:00 | Near-miss: arrival 1 min late → `optimal=0` |

Map hubs for small stations (Alken, Asnæs, Trekroner) live in `web/src/network.ts`.

### Train E

Warm-start from Exp C (not D). E contexts average ~5.3k tokens; use 32k
sequence length and batch 4:

```powershell
python -m tinker_dsb_rl.train_sft `
  file_path=sft_data_expansion_5k/journeys_train.jsonl `
  load_checkpoint_path="tinker://4eb1e06b-c1f6-532b-b592-92a9984536a8:train:0/weights/final" `
  max_length=32768 `
  batch_size=4 `
  num_epochs=1 `
  wandb_project=DSB-journey `
  wandb_name=sft-expE-expansion-5k `
  behavior_if_log_dir_exists=delete
```

Evaluate SFT before deciding whether E needs RLVR:

```powershell
python -m tinker_dsb_rl.eval `
  --journeys journeys_expansion.jsonl `
  --experiment E `
  --checkpoint "<E-SFT-SAMPLER-CHECKPOINT>" `
  --limit 100 `
  --out eval_expansion_sft100.json
```

Only run RLVR if SFT leaves meaningful error signal. Long E prompts make RL
substantially more expensive, so start with smaller rollout batches:

```powershell
python -m tinker_dsb_rl.train `
  journeys_path=journeys_expansion_5k.jsonl `
  services_path=all_services.jsonl `
  reward_mode=dense `
  max_tokens=1536 `
  group_size=4 `
  groups_per_batch=8 `
  load_checkpoint_path="<E-SFT-WEIGHTS-CHECKPOINT>" `
  wandb_project=DSB-journey `
  wandb_name=rl-expE-expansion `
  behavior_if_log_dir_exists=delete
```

E retrieval stays the CLI default; the **F RL** sampler (with thinking) is the
default checkpoint. Use `--heuristic-retrieval` only to compare against the old
gold-free hub retriever; use `--checkpoint` with the E RL path for the previous
silent policy.

## Web UI

React (Vite) frontend + FastAPI backend. Defaults: F RL sampler, thinking on,
`max_transfers=3` (UI/API). Leaflet hubs live in `web/src/network.ts`.

**Terminal 1 — API** (repo root, `dsbpy` active, `TINKER_API_KEY` set):

```powershell
cd C:\Users\I747069\Downloads\Tinker_Køreplan
.\dsbpy\Scripts\Activate.ps1
python -m tinker_dsb_rl.web_api
```

Listens on http://127.0.0.1:8765

**Terminal 2 — frontend:**

```powershell
cd C:\Users\I747069\Downloads\Tinker_Køreplan\web
npx vite --host 127.0.0.1 --port 5173
```

Open http://127.0.0.1:5173/ — Vite proxies `/api` to the backend.

Engine modes: **LLM + RAPTOR**, **Kun RAPTOR**, or **Kun LLM**. RAPTOR-only
does not need Tinker. The LLM tab shows reasoning when thinking is enabled.

### RAPTOR reconstruction correction

The E coverage audit exposed a verifier defect: station-level parent links
could be overwritten in later rounds, so reconstruction occasionally returned
more transfers than `max_transfers`. `Timetable.earliest_journey()` now stores
an immutable path snapshot for each boarding event. The final 100-example E
coverage run contains only 0-, 1-, and 2-transfer journeys.

## Files

| File | Role |
|------|------|
| `tinker_dsb_rl/journey_env.py` | Tinker `ProblemEnv` + dataset builder |
| `tinker_dsb_rl/journey_grading.py` | Parse responses, sparse/dense RLVR reward |
| `tinker_dsb_rl/timetable_context.py` | Retrieve + format local timetable context (A/B/C/E) |
| `tinker_dsb_rl/train.py` | RL training CLI (`reward_mode=`) |
| `tinker_dsb_rl/train_sft.py` | SFT warm-start CLI |
| `tinker_dsb_rl/export_sft_splits.py` | Export chat JSONL for SFT |
| `tinker_dsb_rl/ask.py` | Inference CLI (retrieve → sample → verify) |
| `tinker_dsb_rl/eval.py` | Offline held-out test eval (WandB-compatible metrics) |
| `tinker_dsb_rl/ranking_env.py` | Experiment D ranking reward + RL dataset |
| `tinker_dsb_rl/train_ranking.py` | Experiment D RLVR training CLI |
| `tinker_dsb_rl/eval_ranking.py` | Experiment D held-out evaluation |
| `tinker_dsb_rl/eval_expansion.py` | Experiment E context-coverage evaluation |
| `tinker_dsb_rl/alternatives.py` | RAPTOR via-diverse alternative journeys |
| `tinker_dsb_rl/export_reasoning_sft.py` | Experiment F reasoning SFT export |
| `tinker_dsb_rl/web_api.py` | FastAPI for React UI (RAPTOR + LLM + thinking; `max_transfers` default 3) |
| `web/` | Vite React UI — Leaflet map + routes + reasoning |
| `web/src/network.ts` | Hub coordinates + corridors (incl. Alken, Asnæs, Trekroner) |
| `eval_test100.json` | Offline eval results (100 test ODs × A/B/C) |
| `eval_expansion_rl100.json` | Experiment E RL held-out (100) |
| `eval_expansion_f_sft100.json` | Experiment F SFT held-out (100, thinking) |
| `eval_expansion_f_rl100.json` | Experiment F RL held-out (100, thinking; production) |
| `dsb_routing.py` | RAPTOR router (verifier) |
| `generate_journeys.py` | `generate-direct` (A), `generate-transfer1` (B), `generate-transfer2` (C) |
| `generate_ranking.py` | Experiment D candidate-ranking generator |
| `generate_expansion.py` | Experiment E deterministic-expansion dataset generator |
| `journeys.jsonl` | Original multi-transfer facit (no context) |
| `journeys_direct.jsonl` | Experiment A 0-transfer + context |
| `journeys_transfer1.jsonl` | Experiment B 1-transfer + context |
| `journeys_transfer2.jsonl` | Experiment C 2-transfer + context |

## References

- [Tinker Cookbook — Math RL](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/math_rl)
- [Tinker RL docs](https://tinker-docs.thinkingmachines.ai/cookbook/recipes/)
- [Verifiers RL recipe](https://github.com/thinking-machines-lab/tinker-cookbook/tree/main/tinker_cookbook/recipes/verifiers_rl)
