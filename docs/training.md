# Training and implementation contract

This repository provides a small, standalone reference objective and training
launchers for [NVIDIA-NeMo/labs-molt](https://github.com/NVIDIA-NeMo/labs-molt).
Molt owns asynchronous rollout collection, actual vLLM sampling log probabilities,
model loading, FSDP2, gradient accumulation, evaluation and checkpointing. The
reference objective is for inspection and CPU validation; the launcher executes
Molt's existing implementation rather than monkey-patching its trainer.

## CPU reference

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
python examples/toy_update.py
python -m pytest -q
```

```python
from flashreinforce import flashreinforce_loss

# logps and behavior_logps: [B, T]; rewards: [B]; action_mask: bool [B, T].
# logps retains the current model's autograd graph.
loss, stats = flashreinforce_loss(logps, behavior_logps, rewards, action_mask,
                                delta=3e-3)
optimizer.zero_grad()
loss.backward()
optimizer.step()
# Discard this batch. Obtain a new batch before the next optimizer step.
```

One row must contain the entire trajectory, including all assistant turns. Prompt,
observation, tool-result and padding slots have `action_mask=False`. The full
batch reward mean is computed before admission. Rejected trajectories remain in
B; masked tokens remain in the original policy-token length T_i. A homogeneous
reward batch has zero policy gradient; AdamW momentum/weight decay can still move
parameters if the caller performs an optimizer step.

The function expects finite nonpositive log probabilities on policy tokens and
ignores arbitrary values in non-action slots. It promotes half precision to
float32. The binary-KL calculation clamps probabilities to [1e-6, 1-1e-6] as Molt
does, so this numerical gate differs from exact KL at extreme probabilities.
Importance ratios are detached and not clipped; overflow on admitted tokens
raises an error. The gate and reward baseline are also detached.

Optional appendix filtering is available with `negative_token_fraction`,
`entropies` and explicit boolean `failures`. Entropies are full categorical
entropies, not the surprisal of the sampled token. Successful rows keep all
policy tokens; failed rows keep ceil(q*T_i) highest-entropy tokens. Ties are broken
by token position. This option is implemented in the reference function only;
the launchers use q=1.

The reference function expects the **entire learner batch on one device**. Do not
invoke it independently on each distributed rank or microbatch: that changes the
baseline and weighting. Use Molt for distributed training.

## Install the pinned training backend

On a Linux machine with NVIDIA GPUs, use the installation/container instructions
from Molt at the revision below. A macOS CPU smoke test cannot validate its CUDA
stack. Keep this checkout separate from any existing working checkout:

```bash
git clone https://github.com/NVIDIA-NeMo/labs-molt.git labs-molt-flash
git -C labs-molt-flash checkout --detach 7e796e4e2648f905ae2a44dfc1b2ab98b07b68c3
export MOLT_PATH="$(pwd)/labs-molt-flash"
# Install Molt's dependencies in the training environment, following its README.
pip install -e "$MOLT_PATH"
```

The launcher checks the commit and tracked-file cleanliness before training.
`--dry-run` only prints the command, so it works without Molt, CUDA or datasets.
It does not automatically install GPU dependencies, download checkpoints or
start/stop a shared Ray cluster.

## DeepSeek-R1-Distill-Qwen-1.5B

Prepare the public sanity-test data using Molt's converter:

```bash
python "$MOLT_PATH/examples/python/utils/prepare_dapo.py" \
  --train-source sail/Sanity-Test-R1D-1.5B \
  --eval-source sail/Sanity-Test-R1D-1.5B --eval-split test \
  --out-dir "$PWD/data/sanity-r1d"
ray start --head --num-gpus=8 --disable-usage-stats
python scripts/train_molt.py --recipe r1 --molt-path "$MOLT_PATH" \
  --model /path/to/DeepSeek-R1-Distill-Qwen-1.5B \
  --train-data "$PWD/data/sanity-r1d/train" \
  --eval-data "$PWD/data/sanity-r1d/eval" \
  --output "$PWD/outputs/r1" --dry-run
# Remove --dry-run to train after reviewing the command.
```

Default placement is one actor GPU plus seven rollout GPUs. Adjust
`--actor-gpus` and `--rollout-gpus` for available memory; the Ray cluster must
have enough GPUs for their sum. Actor GPU count must divide batch size 128.
Use `--attention te` in a compatible Transformer Engine installation.

The recipe uses B=128, one rollout per prompt, one epoch and a train batch of
128 (microbatches accumulate before the single update), 512 in-flight rollout
requests, queue depth 8, delta=5e-3, bf16, lr=1e-6, weight decay=.1, warmup=.03,
minimum lr ratio=.1, and 8,192 generated / 9,216 total tokens. Evaluation is
AIME24/25 avg@32, temperature .6, top-p .95, every 128 updates including step 0.
`--delta inf` is the no-sequence-trust ablation.

`--episodes` counts dataset passes, **not optimizer steps**. These are fresh-start
recipes. The paper's extended schedule and fresh-AdamW continuation from update
5,000 are not reconstructed by this launcher; matching those results requires
the original checkpoint and schedule history. Queue depth does not guarantee
policy lag four: measure lag under the actual hardware and workload.

## Qwen2.5-Math-1.5B

```bash
python scripts/train_molt.py --recipe qwen_math --molt-path "$MOLT_PATH" \
  --model /path/to/Qwen2.5-Math-1.5B \
  --train-data /path/to/deduplicated-7500-dapo/train \
  --eval-data /path/to/five-benchmark-eval \
  --output "$PWD/outputs/qwen-math" --dry-run
```

This uses delta=3e-3, 4,096 generated / 6,144 total tokens, B=128,
train temperature/top-p=1, and evaluation avg@16 at temperature/top-p=.7.
Use `--delta 1e-3` for the alternative arithmetic-mean gate configuration.
Queue settings are inherited from the R1 quick start as a runnable starting
point, not a recovered Qwen collection schedule. The 100-update evaluation
interval is a launcher choice.

Supply the paper's deduplicated 7.5k training selection and five benchmark eval
set (MATH-500, AMC23, Minerva, AIME25, OlympiadBench) in Molt's saved-dataset format:
`prompt` chat messages, `reward_model` including `ground_truth`, and a
`datasource` identifying each benchmark. Molt's generic DAPO preparation script
alone supplies AIME24 evaluation and an arbitrary deduplicated training prefix;
it does **not** recover the paper's exact dataset selection or five-task suite.
No missing experiment artifacts or metrics are synthesized here.

## Paper-to-code mapping

This implementation follows the [public paper](../FlashREINFORCE.pdf)
and the public Molt implementation pinned above.

| Paper component | Reference | Molt setting / source |
| --- | --- | --- |
| Batch reward mean | `advantages = rewards - rewards.mean()` | `algo.advantage.estimator=flash_reinforce`, `molt/trainer/algorithm/advantage.py` |
| Token pi/mu | detached exp(logps - behavior_logps) | `train.force_on_policy`, behavior correction in `molt/models/loss.py` |
| Mean Bernoulli KL gate | `sequence_kl <= delta` | `is_correction_level=seq`, `is_correction_gating=binary_kl`, `is_correction_mode=mask` |
| Detached-ratio loss | `flashreinforce/loss.py` | PPO evaluated with old=current.detach(), then detached behavior IS; equal gradient, different scalar loss |
| Sample mean | original T_i then original B | `actor.loss_agg_mode=seq-mean-token-mean` |
| One fresh-batch update | caller's batch lifecycle | samples/prompt=1, rollout batch=train batch=128, max_epochs=1 |
| No reference KL | no reference model | `algo.kl.init_coef=0` |
| Optional entropy selector | explicit failure labels, ceil(q*T_i) | reference only; launchers use q=1 |

Molt additionally limits log-ratios to [-30, 30] as a numerical guard. The
unclipped reference and Molt agree away from this guard; neither uses PPO ratio
clipping to constrain the behavior ratio in these recipes. Passing the sampled
binary-KL gate does not certify small full-vocabulary KL or a return guarantee.

Run the optional upstream gradient comparison on CPU, without importing the
GPU runtime:

```bash
MOLT_PATH=/path/to/pinned/labs-molt-flash python -m pytest -q
```

The optional tests execute Molt's loss module and its `masked_mean` function in
isolation and compare gradients and admission rates, including unequal lengths
and rejected sequences. They do not test distributed scheduling or GPU kernels.

## Additional experiment settings

See [examples/README.md](../examples/README.md) for R1 and Qwen Math variants,
7B Python-tool settings, a 20-turn Qwen3 MoE trust ablation, and ALFWorld settings.
These use the native FlashREINFORCE interface and require a compatible trainer;
the public Molt launcher above retains its pinned interface. Tool/ALFWorld
agents, data and checkpoints are supplied separately. Routing-replay recipes
and staged checkpoint continuation are not included.

Paper scores in the README are reported research results, not outputs of the
CPU example or newly validated reproductions. GPU training and benchmark
reproduction remain to be run on the target infrastructure.
