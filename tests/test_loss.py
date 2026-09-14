import math

import pytest
import torch

from flashreinforce import flashreinforce_loss


def test_gradient_matches_paper_and_detaches_behavior_rewards():
    current = torch.tensor([[-1., -2., -3.], [-2., -1., -2.]], dtype=torch.float64, requires_grad=True)
    behavior = (current.detach() - .1).requires_grad_()
    rewards = torch.tensor([1., 0.], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, False], [True, True, True]])
    loss, stats = flashreinforce_loss(current, behavior, rewards, mask, delta=math.inf)
    loss.backward()
    expected = -(torch.tensor([.5, -.5])[:, None] * math.exp(.1) * mask / mask.sum(-1)[:, None]) / 2
    torch.testing.assert_close(current.grad, expected.to(torch.float64), rtol=1e-6, atol=1e-8)
    assert behavior.grad is None and rewards.grad is None
    assert all(not value.requires_grad for value in stats.values())


def test_sequence_gate_removes_whole_row_keeps_original_batch_denominator():
    current = torch.tensor([[-1., -1.], [-1., -1.]], requires_grad=True)
    behavior = torch.tensor([[-4., -1.], [-1., -1.]])
    loss, stats = flashreinforce_loss(current, behavior, torch.tensor([1., 0.]), torch.ones(2, 2, dtype=torch.bool))
    assert stats['admitted'].tolist() == [False, True]
    loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([[0., 0.], [.125, .125]]))


def test_masked_nan_and_inf_do_not_contaminate_loss_or_gradient():
    current = torch.tensor([[-1., float('nan')], [-1., float('inf')]], requires_grad=True)
    behavior = torch.tensor([[-1., float('inf')], [-1., float('nan')]])
    mask = torch.tensor([[True, False], [True, False]])
    loss, _ = flashreinforce_loss(current, behavior, torch.tensor([1., 0.]), mask)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(current.grad).all()
    assert current.grad[:, 1].eq(0).all()


@pytest.mark.parametrize('fraction,expected_count', [(0., 0), (.2, 1), (.9, 3), (1., 3)])
def test_negative_filter_ceil_and_original_length(fraction, expected_count):
    current = torch.full((2, 4), -1., requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True, True, True, False]])
    loss, stats = flashreinforce_loss(current, current.detach(), torch.tensor([1., 0.]), mask,
        negative_token_fraction=fraction, failures=torch.tensor([False, True]),
        entropies=torch.tensor([[1., 2., 3., 99.], [1., 2., 3., 99.]]))
    loss.backward()
    assert stats['retained_tokens'].tolist() == [3, expected_count]
    assert current.grad[1, 3] == 0
    if expected_count:
        assert current.grad[1, 2].item() == pytest.approx(1 / 12)
    assert current.grad[0, 0].item() == pytest.approx(-1 / 12)


@pytest.mark.parametrize('all_rejected', [False, True])
def test_zero_gradient_for_constant_rewards_or_all_rejected(all_rejected):
    current = torch.full((2, 3), -1., requires_grad=True)
    behavior = torch.full((2, 3), -4. if all_rejected else -1.)
    rewards = torch.tensor([1., 0.] if all_rejected else [1., 1.])
    loss, _ = flashreinforce_loss(current, behavior, rewards, torch.ones(2, 3, dtype=torch.bool))
    loss.backward()
    assert loss.item() == 0 and current.grad.eq(0).all()


def test_binary_kl_direction_and_boundary():
    p, q = .2, .4
    current = torch.tensor([[math.log(q)], [math.log(q)]], dtype=torch.float64)
    behavior = torch.full_like(current, math.log(p))
    args = (current, behavior, torch.tensor([1., 0.]), torch.ones(2, 1, dtype=torch.bool))
    _, stats = flashreinforce_loss(*args, delta=math.inf)
    expected = p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))
    assert stats['sequence_kl'][0].item() == pytest.approx(expected)
    _, boundary = flashreinforce_loss(*args, delta=stats['sequence_kl'][0].item())
    assert boundary['admitted'].all()


@pytest.mark.parametrize('invalid', ['empty_row', 'nan', 'positive', 'fraction', 'delta', 'filter'])
def test_invalid_inputs(invalid):
    current = torch.full((2, 3), -1.)
    mask = torch.ones(2, 3, dtype=torch.bool)
    kwargs = {}
    if invalid == 'empty_row': mask[0] = False
    if invalid == 'nan': current[0, 0] = float('nan')
    if invalid == 'positive': current[0, 0] = .1
    if invalid == 'fraction': kwargs['negative_token_fraction'] = 1.1
    if invalid == 'delta': kwargs['delta'] = -1
    if invalid == 'filter': kwargs['negative_token_fraction'] = .2
    with pytest.raises(ValueError):
        flashreinforce_loss(current, current.clone(), torch.tensor([1., 0.]), mask, **kwargs)


def test_sample_mean_preserves_total_gradient_when_a_trajectory_is_longer():
    totals = []
    for length in (1, 9):
        current = torch.full((2, length), -1., requires_grad=True)
        mask = torch.ones(2, length, dtype=torch.bool)
        mask[0, 1:] = False
        loss, _ = flashreinforce_loss(current, current.detach(), torch.tensor([2., -1.]), mask)
        loss.backward()
        totals.append(current.grad.sum(-1))
    torch.testing.assert_close(totals[0], totals[1])


def test_filter_uses_explicit_failures_not_advantage_sign():
    current = torch.full((2, 2), -1., requires_grad=True)
    loss, stats = flashreinforce_loss(current, current.detach(), torch.tensor([1., 2.]),
        torch.ones(2, 2, dtype=torch.bool), negative_token_fraction=0.,
        failures=torch.tensor([False, True]), entropies=torch.ones(2, 2))
    assert stats['advantages'][0] < 0
    assert stats['retained_tokens'].tolist() == [2, 0]
    loss.backward()
    assert current.grad[0].ne(0).all() and current.grad[1].eq(0).all()


def test_bfloat16_probabilities_promote_before_gate():
    current = torch.zeros(2, 2, dtype=torch.bfloat16, requires_grad=True)
    loss, stats = flashreinforce_loss(current, current.detach(), torch.tensor([1., 0.]),
                                    torch.ones(2, 2, dtype=torch.bool))
    loss.backward()
    assert loss.dtype == torch.float32
    assert stats['admitted'].all() and torch.isfinite(current.grad).all()
