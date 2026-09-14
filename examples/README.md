# Experiment settings

`settings/` contains model, sampling, optimization and evaluation settings for
reasoning, tool use and ALFWorld. Shared flags are in `common.json`; each JSON
file overrides them for one experiment. Named variants override only the listed
flags. Models, datasets, agents and GPU placement are supplied separately.

| Setting | Batch (prompts × samples) | Total / generated tokens | Turns | Sequence gate | Online evaluation |
| --- | --- | --- | --- | --- | --- |
| `r1` | 128 × 1 | 9,216 / 8,192 | 1 | 3e-3; variant 5e-3 | AIME24/25, avg@32, every 128 updates |
| `qwen_math` | 128 × 1 | 6,144 / 4,096 | 1 | 3e-3 | AMC23/Minerva/AIME25, avg@4, every 100 updates |
| `qwen7b_tool` | 128 × 1 | 8,192 / 6,144 | 10 | 1e-3 | AMC23/Minerva/AIME25, avg@4, every 100 updates |
| `qwen3_tool_ablation` | 128 × 1 | 16,384 / 14,336 | 20 | 3e-3 | AMC23/Minerva/AIME25, avg@4, every 100 updates |
| `alfworld` | 64 × 1 | 16,384 / 8,192 | 50 | 1e-3 | 140 seen + 134 unseen games, 3 samples/game, every 40 updates |

Generated-token limits are passed to the trainer's `rollout.max_new_tokens`;
for multi-turn environments the runner also enforces the total context limit.
They should not be interpreted as an independent sum budget over all turns.

The Qwen3 setting above is the **20-turn trust ablation**. It is distinct from
the 10-turn, 18,432-context MoE comparison in the [paper](../FlashREINFORCE.pdf).
The 7B setting describes the 10-turn context budget. Selecting a setting starts
a new run; it does not replay an earlier checkpoint or schedule history.

## Inspect without a GPU or trainer installation

```bash
python examples/run_experiment.py --setting r1 --print-config
python examples/run_experiment.py --setting qwen7b_tool --variant negative_q09 --print-config
python examples/run_experiment.py --setting alfworld --variant grpo --print-config
```

Variant names:

| Setting | Variants |
| --- | --- |
| R1 | `gate_5e3`, `no_trust` |
| Qwen Math | `gate_1e3`, `negative_q09`, `negative_q02`, `five_benchmark_eval16` |
| 7B tool | `negative_q09`, `negative_q02`, `negative_q00`, `token_mean`, `constant_length`, `grpo` |
| Qwen3 tool ablation | `no_trust`, `no_is` |
| ALFWorld | `negative_q02`, `no_is`, `grpo` |

Multiple `--variant` arguments apply in order. The last selected value wins.
The five-benchmark variant changes sampling to avg@16; supply an evaluation
set containing MATH-500, AMC23, Minerva, AIME25 and OlympiadBench as well.
Reduction variants select an objective only; they do not load the continuation
checkpoint used for a reduction study.

Shared FlashREINFORCE settings: one rollout per prompt; one full-batch update;
AdamW lr 1e-6, weight decay .1, max gradient norm 1; no reference KL; sequence-mean
token-mean reduction. Cosine schedules use .03 warmup and .1 minimum-lr ratio.
ALFWorld uses a constant schedule. Entropy filtering keeps exactly ceil(q*T_i)
tokens on failures, without shrinking the T_i denominator; q=1 is the default
except ALFWorld's q=.9. Queue size is 8 for R1, 4 for the Qwen3 ablation, and 1
for Qwen Math, 7B tool and ALFWorld. Queue size is not a guarantee of policy lag.

GRPO variants use 32 × 4 samples for the 7B tool task and 8 × 8 for ALFWorld,
keeping 128 or 64 total trajectories per update. These settings retain the
shared one-pass schedule and use the native trainer's token-mean PPO reduction.
They are configuration comparisons, not a claim to reproduce every GRPO setup.

## Training runtime

These settings use a **native FlashREINFORCE loss** with
`--actor.loss_mode flash_reinforce`, `--actor.flash_neg_topq`,
`--actor.flash_gate_delta`, `--actor.flash_gate_level` and
`--actor.flash_loss_agg`. This loss computes its own behavior correction, so
`--algo.advantage.is_correction_level off` prevents double correction. The
`no_is` variant also disables the behavior-based gate by making the denominator
the detached current policy.

The public Molt revision pinned by [scripts/train_molt.py](../scripts/train_molt.py)
uses a different configuration interface and does not provide these native loss
flags. Use that script for the public Molt base-method examples. To launch the
settings here, provide a trainer installation implementing the native interface.
The launcher runs its `--help` and rejects unsupported flags before starting
training. It does not patch the trainer or silently remove unsupported settings.

```bash
python examples/run_experiment.py --setting qwen_math \
  --trainer-path /path/to/compatible-trainer \
  --model /path/to/Qwen2.5-Math-1.5B \
  --train-data /path/to/dapo-7500/train --eval-data /path/to/math-eval \
  --actor-gpus 2 --rollout-engines 14 \
  --output outputs/qwen-math --dry-run

python examples/run_experiment.py --setting qwen7b_tool --variant negative_q09 \
  --trainer-path /path/to/compatible-trainer \
  --agent-path /path/to/python_tool_agent.py \
  --train-data /path/to/tool-train --eval-data /path/to/tool-eval \
  --actor-gpus 8 --rollout-engines 4 --rollout-tp 2 \
  --output outputs/qwen7b-tool --dry-run

python examples/run_experiment.py --setting alfworld \
  --trainer-path /path/to/compatible-trainer \
  --agent-path /path/to/alfworld_agent.py \
  --train-data /path/to/alfworld/train --eval-data /path/to/alfworld/eval \
  --actor-gpus 8 --actor-tp 2 --rollout-engines 4 --rollout-tp 2 \
  --output outputs/alfworld --dry-run
```

GPU allocations in these commands are examples. Set `--actor-nodes`,
`--actor-gpus`, `--actor-tp`, `--actor-ep`, `--actor-cp`, `--rollout-engines`
and `--rollout-tp` for the model and cluster. Start Ray separately with enough
resources. The launcher does not start or stop Ray. Remove `--dry-run` to execute;
previews do not validate the installed trainer. A resolved copy of the actual
flags is saved under the chosen output directory when training starts.

Agent/data requirements:

- Math: saved datasets with `prompt`, `reward_model` and benchmark `datasource`.
  The default agent path is the trainer's single-turn math agent. The Qwen Math
  checkpoint configuration must support 6,144-token contexts.
- Python tools: tool-formatted system prompts, a compatible parser/executor,
  final-answer grading and stored behavior log probabilities across all turns.
  Supply the agent explicitly; a text-only math agent cannot substitute for it.
- ALFWorld: install the environment and game data, export `ALFWORLD_DATA` to a
  path available on every worker, and use `game_file` labels. Supply an agent
  that returns terminal environment success and excludes observations from the
  policy-token mask.

Agent implementations, datasets and checkpoints are not bundled. These settings
have command/configuration tests; GPU training and benchmark reproduction have
not been rerun here. `toy_update.py` remains the standalone CPU loss example.
