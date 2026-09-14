"""Launch paper-aligned single-turn recipes against a pinned Molt checkout."""

import argparse
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

MOLT_REVISION = "7e796e4e2648f905ae2a44dfc1b2ab98b07b68c3"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", choices=["r1", "qwen_math"], default="r1")
    parser.add_argument("--molt-path", type=Path, required=True)
    parser.add_argument("--model", required=True, help="Local checkpoint or Hugging Face model ID")
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/flashreinforce"))
    parser.add_argument("--actor-gpus", type=int, default=1)
    parser.add_argument("--rollout-gpus", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=1000,
                        help="Dataset passes, NOT optimizer updates")
    parser.add_argument("--delta", type=float, help="inf selects the no-trust ablation")
    parser.add_argument("--attention", default="flash_attention_2", choices=["flash_attention_2", "te"])
    parser.add_argument("--dry-run", action="store_true", help="Print command without importing Molt or starting training")
    args = parser.parse_args()
    if min(args.actor_gpus, args.rollout_gpus, args.episodes) < 1 or 128 % args.actor_gpus:
        parser.error("GPU counts/episodes must be positive; actor GPUs must divide batch size 128")
    r1 = args.recipe == "r1"
    delta = args.delta if args.delta is not None else (5e-3 if r1 else 3e-3)
    if math.isnan(delta) or delta < 0:
        parser.error("delta must be nonnegative")
    root = args.molt_path.expanduser().resolve()
    agent = root / "examples/python/agents/math.py"
    output = args.output.expanduser().resolve()
    flags = {
        "actor.model_name_or_path": args.model,
        "data.prompt_dataset": args.train_data.expanduser().resolve(),
        "data.input_key": "prompt", "data.label_key": "reward_model",
        "data.max_samples": 1460 if r1 else 7500,
        "data.max_len": 9216 if r1 else 6144,
        "rollout.batch_size": 128, "rollout.vllm_generate_batch_size": 512,
        "rollout.micro_batch_size": 1, "rollout.n_samples_per_prompt": 1,
        "rollout.max_new_tokens": 8192 if r1 else 4096,
        "rollout.temperature": 1.0, "rollout.top_p": 1.0,
        "train.batch_size": 128, "train.micro_batch_size": 1,
        "train.max_epochs": 1, "train.num_episodes": args.episodes,
        "train.async_queue_size": 8,
        "actor.num_nodes": 1, "actor.num_gpus_per_node": args.actor_gpus,
        "ref.num_nodes": 1, "ref.num_gpus_per_node": args.actor_gpus,
        "vllm.num_engines": args.rollout_gpus, "vllm.tensor_parallel_size": 1,
        "vllm.sync_backend": "nccl", "vllm.gpu_memory_utilization": 0.9,
        "fsdp.param_dtype": "bf16", "fsdp.attn_implementation": args.attention,
        "actor.gradient_checkpoint": "full", "actor.optim": "adam",
        "actor.adam.lr": 1e-6, "actor.adam.weight_decay": 0.1,
        "actor.lr_scheduler": "cosine_with_min_lr", "actor.lr_warmup_ratio": 0.03,
        "actor.min_lr_ratio": 0.1, "actor.max_norm": 1.0,
        "actor.loss_agg_mode": "seq-mean-token-mean", "actor.entropy_coef": 0,
        "algo.advantage.estimator": "flash_reinforce",
        "algo.advantage.is_correction_level": "seq",
        "algo.advantage.is_correction_mode": "mask",
        "algo.advantage.is_correction_gating": "binary_kl",
        "algo.advantage.is_correction_threshold": delta, "algo.kl.init_coef": 0,
        "train.agent_path": agent, "eval.dataset": args.eval_data.expanduser().resolve(),
        "eval.steps": 128 if r1 else 100,
        "eval.n_samples_per_prompt": 32 if r1 else 16,
        "eval.temperature": 0.6 if r1 else 0.7, "eval.top_p": 0.95 if r1 else 0.7,
        "ckpt.output_dir": output / "hf", "ckpt.path": output / "state",
        "ckpt.save_steps": 50, "logger.logging_steps": 1,
    }
    command = [sys.executable, "-u", "-m", "molt.cli.train_rl_ray"]
    for key, value in flags.items():
        command.extend([f"--{key}", str(value)])
    command += ["--data.apply_chat_template", "--train.force_on_policy",
                "--train.colocate_fsdp_models", "--fsdp.packing_samples", "--eval.eval_at_start"]
    print(f"Molt revision: {MOLT_REVISION}\nWorking directory: {root}", flush=True)
    print("MAX_AGENT_TURNS=1 " + shlex.join(command), flush=True)
    if args.dry_run:
        return
    if not agent.is_file():
        parser.error(f"Molt math agent not found: {agent}")
    revision = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if revision != MOLT_REVISION:
        parser.error(f"Expected Molt {MOLT_REVISION}; found {revision}. See docs/training.md")
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"], text=True).strip():
        parser.error("Molt has modified tracked files; use a clean pinned checkout")
    for dataset in (args.train_data, args.eval_data):
        if not dataset.expanduser().exists():
            parser.error(f"Prepared dataset does not exist: {dataset}")
    env = os.environ.copy()
    env.update(MAX_AGENT_TURNS="1", TOKENIZERS_PARALLELISM="true", RAY_USAGE_STATS_ENABLED="0",
               VLLM_WORKER_MULTIPROC_METHOD="spawn")
    # Ray's lifecycle belongs to the caller; do not stop a shared cluster here.
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == "__main__":
    main()
