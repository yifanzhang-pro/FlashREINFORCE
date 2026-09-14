# FlashREINFORCE

[![Paper](https://img.shields.io/badge/Paper-b31b1b.svg)](./FlashREINFORCE.pdf)
[![Website](https://img.shields.io/badge/Project-Website-blue)](https://yifanzhang-pro.github.io/FlashREINFORCE/)
[![Code](https://img.shields.io/badge/Code-NVIDIA--NeMo%2Flabs--molt-76B900)](https://github.com/NVIDIA-NeMo/labs-molt)
[![License](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](./LICENSE)

### Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models

**FlashREINFORCE** is a critic-free, one-pass reinforcement-learning framework for asynchronous agent training. It uses **one rollout per prompt**, corrects stale behavior with token importance sampling, controls accumulated policy drift with a **Sequence Trust Region**, and prevents long failed trajectories from dominating optimization with **Sample-Mean Optimization**.

> **Reinforcement Learning Should Do REINFORCE.**

**Authors:** Jian Hu*, Yifan Zhang*, Hao Zhang*, Binfeng Xu*, Shaokun Zhang, Hongqing Peng, Zhiding Yu, Pavlo Molchanov, Jan Kautz, Yi Dong  
**Affiliation:** NVIDIA  
**Date:** September 2026

[[Paper](./FlashREINFORCE.pdf)] [[Project Website](https://yifanzhang-pro.github.io/FlashREINFORCE/)] [[Code](https://github.com/NVIDIA-NeMo/labs-molt)]

## Code and quick start

This repository now includes a standalone PyTorch reference loss, a CPU update
example, pinned Molt launchers for R1 and Qwen2.5-Math, and experiment settings
for reasoning, Python tools, MoE and ALFWorld. The launchers use
Molt's asynchronous training implementation.

```bash
pip install -e '.[test]'
python examples/toy_update.py
python -m pytest -q
```

- [Reference objective](flashreinforce/loss.py): batch centering, token IS,
  sequence trust, sample mean, and optional entropy-based failure-token filtering.
- [Training guide](docs/training.md): GPU setup, data preparation, commands,
  paper-to-code mapping, and reproduction limitations.
- [Molt launcher](scripts/train_molt.py): `--dry-run` previews all training flags.
- [Experiment settings](examples/README.md): task presets, ablations and a native
  trainer launcher with capability checks.

The CPU example checks an optimizer update; it does not reproduce the paper's
benchmark scores. Tool and ALFWorld settings require compatible agents and
environments; see the examples guide for runtime requirements.

## Why single-rollout RL?

Long-horizon agents have irregular rollout times because of tool calls, environment interaction, and variable-length reasoning. Group-relative methods require multiple sibling rollouts of the same prompt, which reduces prompt coverage at a fixed rollout budget and introduces synchronization barriers.

FlashREINFORCE instead uses **one rollout per prompt**. A batch of $B$ completed trajectories can therefore cover $B$ distinct prompts, and each trajectory can enter learning as soon as it finishes.

The central challenge is stability: without sibling rollouts there is no group baseline, asynchronous trajectories can be stale, and long failures can receive disproportionately large negative updates. FlashREINFORCE addresses these issues with three components.

## Three components

### 1. One-Batch REINFORCE

For a fresh batch of completed trajectories, FlashREINFORCE centers scalar rewards across independent prompts:

```math
\bar R=\frac{1}{B}\sum_{j=1}^{B}R_j,
\qquad
A_i=R_i-\bar R.
```

This gives **signed feedback without a critic**: above-mean trajectories receive positive advantage and below-mean trajectories receive negative advantage. Each fresh batch receives one optimizer update and is then discarded.

### 2. Sequence Trust Region

For token $t$ in trajectory $i$, let $\mu_i$ be the behavior policy that generated the action and $\pi_\theta$ the current learner. FlashREINFORCE stores the actual behavior probability and uses the token importance ratio

```math
\rho_{i,t}(\theta)
=\exp\!\left(
\log\pi_\theta(a_{i,t}\mid h_{i,t})
-\log\mu_i(a_{i,t}\mid h_{i,t})
\right).
```

Token importance sampling corrects the conditional action distribution at the stored history, but the stored histories themselves still come from the behavior policy. FlashREINFORCE therefore also screens accumulated trajectory-level drift.

For the sampled action, define the Bernoulli KL proxy

```math
d_{i,t}
=p_{i,t}\log\frac{p_{i,t}}{q_{i,t}}
+(1-p_{i,t})\log\frac{1-p_{i,t}}{1-q_{i,t}},
```

where $p_{i,t}=\mu_i(a_{i,t}\mid h_{i,t})$ and $q_{i,t}=\pi_\theta(a_{i,t}\mid h_{i,t})$. The sequence statistic and admission mask are

```math
\bar D_i=\frac1{T_i}\sum_{t=1}^{T_i}d_{i,t},
\qquad
m_i=\mathbf 1[\bar D_i\le\delta].
```

A rejected trajectory contributes no token losses, giving one trust decision for the complete stored history.

### 3. Sample-Mean Optimization

A failed rollout can contain many useful intermediate steps. If losses are averaged over all tokens in the batch, a long failure receives more weight simply because it is long. FlashREINFORCE instead averages **within each trajectory first**, then across trajectories:

```math
\widehat{\mathcal J}_{\mathrm{FlashREINFORCE}}(\theta)
=\frac1B\sum_{i=1}^{B}\frac{m_iA_i}{T_i}
\sum_{t=1}^{T_i}
\frac{\pi_\theta(a_{i,t}\mid h_{i,t})}
     {\mu_i(a_{i,t}\mid h_{i,t})}.
```

The corresponding update direction is

```math
\widehat g(\theta)
=\frac1B\sum_{i=1}^{B}\frac{m_iA_i}{T_i}
\sum_{t=1}^{T_i}\rho_{i,t}(\theta)
\nabla_\theta\log\pi_\theta(a_{i,t}\mid h_{i,t}).
```

The practical loss uses detached importance ratios. The base method uses **no learned critic, no ratio clipping, and no reference-model forward pass**.

## One-pass asynchronous update

1. **Collect:** rollout workers independently submit one completed trajectory per prompt, together with the behavior-policy probabilities that actually generated its tokens.
2. **Batch:** the learner takes the next $B$ completed trajectories and computes $A_i=R_i-\bar R$.
3. **Correct:** recompute current-policy log-probabilities and form token importance ratios $\rho_{i,t}$.
4. **Trust:** compute the mean sampled-action KL proxy $\bar D_i$ and admit the complete trajectory iff $\bar D_i\le\delta$.
5. **Optimize:** take one sample-mean optimizer step.
6. **Discard:** never replay the same collected batch for additional learner updates.

## Main results

| Setting | FlashREINFORCE result |
| --- | --- |
| **DeepSeek-R1-Distill-Qwen-1.5B, long-CoT** | Stable through **6,000 updates** at policy lag $\approx4$; AIME24/25 mean improves from **21.7 to 33.7**. |
| **Qwen2.5-Math-1.5B** | **38.0** mean accuracy across MATH-500, AMC23, Minerva, AIME25, and OlympiadBench, versus **36.3** for the reported GRPO baseline, using **256k vs. 512k rollouts**. |
| **Qwen2.5-7B-Instruct + Python tool** | **37.0** three-task mean at step 600 and **3.25 tool calls/trajectory**; the compared GRPO run reaches 30.3 and stops calling the tool. |
| **Qwen3-30B-A3B MoE + Python tool** | Stable asynchronous training at policy lag $\approx8$; FlashREINFORCE leads the matched-budget GRPO comparison by **6.8 points**. |
| **ALFWorld, Qwen2.5-7B-Instruct** | **98.3% seen / 96.5% unseen** success at step 200. |

These experiments span mathematical reasoning, multi-turn tool use, and interactive decision making.

## Design notes

- **Store the real sampling probability.** Recomputing an "old" log-probability later is not guaranteed to reproduce the probability used by the inference engine.
- **Importance sampling needs support.** Exact action correction assumes the target policy does not place probability mass outside the behavior policy's support; aggressive top-$k$/top-$p$ truncation can violate this condition.
- **Token IS is local.** It corrects actions at stored histories, not the distribution of those histories. The Sequence Trust Region is designed to control the remaining accumulated drift.
- **Fresh batches matter.** FlashREINFORCE performs one full-batch update per collected batch instead of processing several sequential minibatches from the same behavior snapshot.
- **Sample mean is intentional.** Each trajectory receives weight $1/B$ regardless of response length, preventing long failures from receiving an automatic length multiplier.

## Resources

- [Paper](./FlashREINFORCE.pdf)
- [Project website](https://yifanzhang-pro.github.io/FlashREINFORCE/)
- [Code: NVIDIA-NeMo/labs-molt](https://github.com/NVIDIA-NeMo/labs-molt)

## Citation

```bibtex
@article{hu2026flashreinforce,
  title   = {FlashREINFORCE: Critic-Free Single-Rollout Asynchronous RL for Agentic Language Models},
  author  = {Hu, Jian and Zhang, Yifan and Zhang, Hao and Xu, Binfeng and Zhang, Shaokun and Peng, Hongqing and Yu, Zhiding and Molchanov, Pavlo and Kautz, Jan and Dong, Yi},
  year    = {2026},
  url     = {https://github.com/yifanzhang-pro/FlashREINFORCE}
}
```

## License

Licensed under the [Apache License 2.0](./LICENSE).
