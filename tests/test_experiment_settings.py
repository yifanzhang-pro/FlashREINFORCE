import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('experiment_settings', ROOT / 'examples/run_experiment.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CASES = []
for path in (ROOT / 'examples/settings').glob('*.json'):
    if path.stem != 'common':
        variants = json.loads(path.read_text())['variants']
        CASES.extend((path.stem, variant) for variant in [None, *variants])


@pytest.mark.parametrize('name,variant', CASES)
def test_all_settings_preserve_batch_and_loss_contract(name, variant):
    config = MODULE.load_settings(name, [variant] if variant else [])
    flags = config['flags']
    assert flags['rollout.batch_size'] * flags['rollout.n_samples_per_prompt'] == flags['train.batch_size']
    assert flags['train.max_epochs'] == 1 and flags['train.force_on_policy']
    assert flags['algo.kl.init_coef'] == 0
    assert flags['data.max_len'] > flags['rollout.max_new_tokens']
    if flags['actor.loss_mode'] == 'flash_reinforce':
        assert flags['algo.advantage.is_correction_level'] == 'off'
        assert flags['rollout.n_samples_per_prompt'] == 1
        assert 0 <= flags['actor.flash_neg_topq'] <= 1
    else:
        assert not any(key.startswith('actor.flash_') for key in flags)
        assert flags['algo.advantage.estimator'] == 'grpo'


def test_settings_distinguish_collection_and_online_evaluation():
    math = MODULE.load_settings('qwen_math')['flags']
    r1 = MODULE.load_settings('r1')['flags']
    assert (math['rollout.vllm_generate_batch_size'], math['train.async_queue_size']) == (16, 1)
    assert (r1['rollout.vllm_generate_batch_size'], r1['train.async_queue_size']) == (512, 8)
    assert math['eval.n_samples_per_prompt'] == 4
    assert MODULE.load_settings('qwen_math', ['five_benchmark_eval16'])['flags']['eval.n_samples_per_prompt'] == 16
    assert MODULE.load_settings('qwen7b_tool')['max_agent_turns'] == 10
    assert MODULE.load_settings('qwen3_tool_ablation')['max_agent_turns'] == 20


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match='Unknown variant'):
        MODULE.load_settings('r1', ['typo'])


@pytest.mark.parametrize('supported', [False, True])
def test_launch_checks_runtime_flags_before_training(tmp_path, supported):
    trainer = tmp_path / 'trainer with spaces'
    cli = trainer / 'molt/cli'
    cli.mkdir(parents=True)
    agent = trainer / 'examples/python/agents/math.py'
    agent.parent.mkdir(parents=True)
    agent.write_text('')
    data = tmp_path / 'data'
    data.mkdir()
    output = tmp_path / 'output'
    command = [sys.executable, str(ROOT / 'examples/run_experiment.py'), '--setting', 'qwen_math',
               '--trainer-path', str(trainer), '--train-data', str(data), '--eval-data', str(data),
               '--actor-gpus', '2', '--rollout-engines', '2', '--output', str(output)]
    preview = subprocess.run([*command, '--dry-run'], text=True, capture_output=True, check=True)
    import re
    flag_names = set(re.findall(r'--[A-Za-z0-9_.-]+', preview.stdout))
    if not supported:
        flag_names.discard('--actor.flash_neg_topq')
    (cli / 'train_rl_ray.py').write_text(
        'import os, sys\nfrom pathlib import Path\n'
        f'if "--help" in sys.argv: print({" ".join(sorted(flag_names))!r})\n'
        'else: Path("started").write_text(os.environ["MAX_AGENT_TURNS"])\n')
    result = subprocess.run(command, text=True, capture_output=True)
    if supported:
        assert result.returncode == 0, result.stderr
        assert (trainer / 'started').read_text() == '1'
        resolved = json.loads((output / 'resolved_settings.json').read_text())
        assert resolved['flags']['train.agent_path'] == str(agent)
    else:
        assert result.returncode != 0
        assert 'does not support' in result.stderr
        assert '--actor.flash_neg_topq' in result.stderr
        assert not (trainer / 'started').exists()
        assert not output.exists()


def test_tool_setting_requires_agent_even_for_preview(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / 'examples/run_experiment.py'),
        '--setting', 'qwen7b_tool', '--trainer-path', str(tmp_path),
        '--train-data', str(tmp_path), '--eval-data', str(tmp_path),
        '--actor-gpus', '1', '--rollout-engines', '1', '--dry-run'], capture_output=True, text=True)
    assert result.returncode != 0 and '--agent-path' in result.stderr
