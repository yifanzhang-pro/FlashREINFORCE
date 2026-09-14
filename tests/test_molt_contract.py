"""Optional CPU checks against source from the pinned Molt checkout.

MOLT_PATH=/path/to/labs-molt python -m pytest tests/test_molt_contract.py -q
Loads only the loss module and masked_mean, without Ray/vLLM/AutoModel imports.
This checks gradient compatibility, not the distributed training runtime.
"""
import ast
import os
from pathlib import Path
import subprocess
import sys
from typing import Optional

import pytest
import torch

from flashreinforce import flashreinforce_loss

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('recipe,delta,length', [('r1', '0.005', '8192'), ('qwen_math', '0.003', '4096')])
def test_launcher_dry_run(recipe, delta, length):
    result = subprocess.run([sys.executable, str(ROOT / 'scripts/train_molt.py'),
        '--recipe', recipe, '--molt-path', '/tmp/molt path', '--model', 'model',
        '--train-data', '/tmp/train', '--eval-data', '/tmp/eval', '--dry-run'],
        capture_output=True, text=True, check=True)
    command = result.stdout
    for flag in ['--train.force_on_policy', '--train.max_epochs 1', '--train.batch_size 128',
                 '--rollout.batch_size 128', '--rollout.n_samples_per_prompt 1',
                 '--algo.advantage.estimator flash_reinforce',
                 '--actor.loss_agg_mode seq-mean-token-mean',
                 f'--algo.advantage.is_correction_threshold {delta}',
                 f'--rollout.max_new_tokens {length}']:
        assert flag in command


@pytest.mark.parametrize('delta', [.003, float('inf')])
def test_gradient_matches_pinned_molt(delta):
    root = os.environ.get('MOLT_PATH')
    if root is None:
        pytest.skip('Set MOLT_PATH to the pinned checkout for upstream compatibility checks')
    root = Path(root)
    expected = '7e796e4e2648f905ae2a44dfc1b2ab98b07b68c3'
    assert subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip() == expected
    namespace = {'torch': torch, 'Optional': Optional}
    utils = ast.parse((root / 'molt/models/utils.py').read_text())
    masked_mean = next(node for node in utils.body if isinstance(node, ast.FunctionDef) and node.name == 'masked_mean')
    exec(compile(ast.Module(body=[masked_mean], type_ignores=[]), 'molt_masked_mean', 'exec'), namespace)
    loss_tree = ast.parse((root / 'molt/models/loss.py').read_text())
    loss_tree.body = [node for node in loss_tree.body if not (isinstance(node, ast.ImportFrom) and node.level)]
    exec(compile(loss_tree, 'molt_loss', 'exec'), namespace)
    current = torch.tensor([[-1., -2., -3.], [-2., -1., -4.], [-1., -1., -1.]], requires_grad=True)
    behavior = current.detach().clone() - .01
    behavior[2, 0] = -5.
    rewards = torch.tensor([1., 0., 0.])
    mask = torch.tensor([[True, False, False], [True, True, True], [True, True, False]])
    reference_loss, stats = flashreinforce_loss(current, behavior, rewards, mask, delta=delta)
    reference_gradient, = torch.autograd.grad(reference_loss, current)
    policy_loss = namespace['PolicyLoss'](
        is_correction_threshold=[-float('inf'), delta], is_correction_level='seq',
        is_correction_gating='binary_kl', is_correction_mode='mask',
        loss_agg_mode='seq-mean-token-mean')
    advantage = (rewards - rewards.mean())[:, None].expand_as(current)
    upstream_loss, *_, filtered = policy_loss(current, current.detach(), advantage, mask, behavior)
    upstream_gradient, = torch.autograd.grad(upstream_loss, current)
    torch.testing.assert_close(reference_gradient, upstream_gradient)
    torch.testing.assert_close(1 - filtered, stats['acceptance_rate'])
