"""Inspect experiment settings or launch a trainer with native FlashREINFORCE support."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

SETTINGS = Path(__file__).resolve().parent / 'settings'


def load_settings(name, variants=()):
    """Merge shared flags, one experiment, then explicitly selected variants."""
    config = json.loads((SETTINGS / f'{name}.json').read_text())
    flags = json.loads((SETTINGS / 'common.json').read_text())
    flags.update(config['flags'])
    for variant in variants:
        if variant not in config['variants']:
            raise ValueError(f'Unknown variant {variant!r}; choices: {list(config["variants"])}')
        flags.update(config['variants'][variant])
    if flags['actor.loss_mode'] != 'flash_reinforce':
        flags = {k: v for k, v in flags.items() if not k.startswith('actor.flash_')}
    batch = flags['rollout.batch_size'] * flags['rollout.n_samples_per_prompt']
    if batch != flags['train.batch_size'] or flags['train.max_epochs'] != 1:
        raise ValueError('Each collected batch must receive one full-batch update')
    if flags['actor.loss_mode'] == 'flash_reinforce':
        if flags['rollout.n_samples_per_prompt'] != 1:
            raise ValueError('FlashREINFORCE settings require one rollout per prompt')
        if flags['algo.advantage.is_correction_level'] != 'off':
            raise ValueError('Native FlashREINFORCE already applies behavior correction')
    return {**config, 'flags': flags, 'selected_variants': list(variants)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = sorted(p.stem for p in SETTINGS.glob('*.json') if p.stem != 'common')
    parser.add_argument('--setting', choices=names, required=True)
    parser.add_argument('--variant', action='append', default=[])
    parser.add_argument('--print-config', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--trainer-path', type=Path)
    parser.add_argument('--agent-path', type=Path, help='Required for Python-tool and ALFWorld settings')
    parser.add_argument('--model', help='Override the model ID with a compatible local checkpoint')
    parser.add_argument('--train-data', type=Path)
    parser.add_argument('--eval-data', type=Path)
    parser.add_argument('--output', type=Path, default=Path('outputs/experiment'))
    parser.add_argument('--actor-nodes', type=int, default=1)
    parser.add_argument('--actor-gpus', type=int)
    parser.add_argument('--rollout-engines', type=int)
    parser.add_argument('--rollout-tp', type=int, default=1)
    parser.add_argument('--actor-tp', type=int, default=1)
    parser.add_argument('--actor-ep', type=int, default=1)
    parser.add_argument('--actor-cp', type=int, default=1)
    parser.add_argument('--attention', choices=['flash_attention_2', 'te'], default='flash_attention_2')
    args = parser.parse_args()
    try:
        config = load_settings(args.setting, args.variant)
    except ValueError as exc:
        parser.error(str(exc))
    if args.print_config:
        print(json.dumps(config, indent=2))
        return
    for required in ('trainer_path', 'train_data', 'eval_data', 'actor_gpus', 'rollout_engines'):
        if getattr(args, required) is None:
            parser.error(f'--{required.replace("_", "-")} is required to build a command')
    if min(args.actor_nodes, args.actor_gpus, args.rollout_engines, args.rollout_tp,
           args.actor_tp, args.actor_ep, args.actor_cp) < 1:
        parser.error('GPU and parallelism counts must be positive')
    root = args.trainer_path.expanduser().resolve()
    agent = args.agent_path
    if agent is None:
        if config['agent_kind'] != 'math':
            parser.error('This setting requires --agent-path with a compatible environment agent')
        agent = root / 'examples/python/agents/math.py'
    agent = agent.expanduser().resolve()
    output = args.output.expanduser().resolve()
    flags = dict(config['flags'])
    flags.update({
        'actor.model_name_or_path': args.model or config['model'],
        'train.agent_path': str(agent), 'data.prompt_dataset': str(args.train_data.expanduser().resolve()),
        'eval.dataset': str(args.eval_data.expanduser().resolve()),
        'actor.num_nodes': args.actor_nodes, 'actor.num_gpus_per_node': args.actor_gpus,
        'ref.num_nodes': args.actor_nodes, 'ref.num_gpus_per_node': args.actor_gpus,
        'fsdp.tp_size': args.actor_tp, 'fsdp.ep_size': args.actor_ep, 'fsdp.cp_size': args.actor_cp,
        'vllm.num_engines': args.rollout_engines, 'vllm.tensor_parallel_size': args.rollout_tp,
        'fsdp.attn_implementation': args.attention,
        'ckpt.output_dir': str(output / 'hf'), 'ckpt.path': str(output / 'state'),
    })
    command = [sys.executable, '-u', '-m', 'molt.cli.train_rl_ray']
    for key, value in flags.items():
        if value is False or value is None:
            continue
        command.append(f'--{key}')
        if value is not True:
            command.extend(str(x) for x in (value if isinstance(value, list) else [value]))
    print(f'Working directory: {root}', flush=True)
    print(f'MAX_AGENT_TURNS={config["max_agent_turns"]} ' + shlex.join(command), flush=True)
    if args.dry_run:
        print('Command preview only; trainer capabilities and environment have not been checked.')
        return
    for path in (root / 'molt/cli/train_rl_ray.py', agent,
                 args.train_data.expanduser(), args.eval_data.expanduser()):
        if not path.exists():
            parser.error(f'Required file or dataset not found: {path}')
    if config['agent_kind'] == 'alfworld' and not os.environ.get('ALFWORLD_DATA'):
        parser.error('Set ALFWORLD_DATA to the game-data directory available to all workers')
    env = os.environ.copy()
    env.update(MAX_AGENT_TURNS=str(config['max_agent_turns']), RAY_USAGE_STATS_ENABLED='0',
               VLLM_WORKER_MULTIPROC_METHOD='spawn', TOKENIZERS_PARALLELISM='true')
    # Ask the selected runtime itself; do not silently drop unsupported options.
    help_result = subprocess.run([sys.executable, '-m', 'molt.cli.train_rl_ray', '--help'],
                                 cwd=root, env=env, capture_output=True, text=True)
    if help_result.returncode:
        parser.error('Trainer capability check failed:\n' + help_result.stderr[-3000:])
    available = set(re.findall(r'--[A-Za-z0-9_.-]+', help_result.stdout))
    missing = sorted(f'--{key}' for key, value in flags.items()
                     if value is not False and value is not None and f'--{key}' not in available)
    if missing:
        parser.error('Trainer does not support this setting: ' + ', '.join(missing)
                     + '. Native FlashREINFORCE support is required; see examples/README.md.')
    output.mkdir(parents=True, exist_ok=True)
    (output / 'resolved_settings.json').write_text(json.dumps({
        'setting': args.setting, 'variants': args.variant,
        'max_agent_turns': config['max_agent_turns'], 'flags': flags,
    }, indent=2) + '\n')
    subprocess.run(command, cwd=root, env=env, check=True)


if __name__ == '__main__':
    main()
