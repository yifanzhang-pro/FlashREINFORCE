"""Paper's detached-ratio loss, operating on one complete learner batch."""

import math

import torch
from torch import Tensor


def flashreinforce_loss(
    log_probs: Tensor,
    behavior_log_probs: Tensor,
    rewards: Tensor,
    action_mask: Tensor,
    *,
    delta: float = 3e-3,
    negative_token_fraction: float = 1.0,
    entropies: Tensor | None = None,
    failures: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Return a scalar loss and detached diagnostics.

    Each [B, T] row is a COMPLETE trajectory, including all agent turns.
    action_mask is boolean and excludes prompts, tool observations and padding.
    rewards is [B]. Behavior log probabilities must be the actual sampling logps.
    Center rewards over the full batch before any microbatch/DP partitioning.
    This reference function does not perform distributed reductions.

    Optional entropy filtering requires explicit boolean failure labels and full
    categorical token entropies; it keeps ceil(q*T_i) tokens in failed rows.
    The denominators remain the original B and T_i after either filter.
    delta=inf disables sequence rejection; q=1 is the paper's base method.
    """
    if log_probs.ndim != 2 or not log_probs.numel():
        raise ValueError("log_probs must be a nonempty [B, T] tensor")
    if behavior_log_probs.shape != log_probs.shape or action_mask.shape != log_probs.shape:
        raise ValueError("log probabilities and action_mask must share [B, T] shape")
    if rewards.shape != (log_probs.shape[0],):
        raise ValueError("rewards must have shape [B]")
    if action_mask.dtype != torch.bool:
        raise ValueError("action_mask must be boolean")
    if not log_probs.is_floating_point() or not behavior_log_probs.is_floating_point():
        raise ValueError("log probabilities must be floating point")
    if any(t.device != log_probs.device for t in (behavior_log_probs, rewards, action_mask)):
        raise ValueError("all inputs must be on the same device")
    if math.isnan(delta) or delta < 0 or not 0 <= negative_token_fraction <= 1:
        raise ValueError("delta must be nonnegative and negative_token_fraction in [0, 1]")
    lengths = action_mask.sum(-1)
    if (lengths == 0).any():
        raise ValueError("every trajectory must contain at least one policy token")
    if not torch.isfinite(rewards).all():
        raise ValueError("rewards must be finite")
    for values in (log_probs, behavior_log_probs):
        valid = values[action_mask]
        if not torch.isfinite(valid).all() or (valid > 0).any():
            raise ValueError("policy-token log probabilities must be finite and <= 0")

    # Promote low precision before exponentiation; preserve float64 for checks.
    dtype = torch.float64 if log_probs.dtype == torch.float64 else torch.float32
    current = torch.where(action_mask, log_probs.to(dtype), 0.0)
    with torch.no_grad():
        behavior = torch.where(action_mask, behavior_log_probs.to(dtype), 0.0)
        advantages = rewards.to(dtype) - rewards.to(dtype).mean()
        # Match Molt's numerical stabilization for the binary-KL gate only.
        p = behavior.exp().clamp(1e-6, 1 - 1e-6)
        q = current.detach().exp().clamp(1e-6, 1 - 1e-6)
        divergence = p * (p.log() - q.log()) + (1 - p) * (torch.log1p(-p) - torch.log1p(-q))
        sequence_kl = torch.where(action_mask, divergence, 0.0).sum(-1) / lengths
        sequence_kl = sequence_kl.clamp_min(0)
        admitted = sequence_kl <= delta
        token_mask = action_mask.clone()
        if negative_token_fraction < 1:
            if entropies is None or failures is None:
                raise ValueError("entropy filtering requires entropies and explicit failures")
            if entropies.shape != log_probs.shape or failures.shape != rewards.shape:
                raise ValueError("entropies must be [B, T] and failures [B]")
            if failures.dtype != torch.bool or not entropies.is_floating_point():
                raise ValueError("failures must be boolean and entropies floating point")
            if entropies.device != log_probs.device or failures.device != log_probs.device:
                raise ValueError("filter inputs must be on the same device")
            if not torch.isfinite(entropies[action_mask]).all():
                raise ValueError("policy-token entropies must be finite")
            for i in torch.where(failures)[0].tolist():
                count = math.ceil(negative_token_fraction * lengths[i].item())
                scores = entropies[i].masked_fill(~action_mask[i], -torch.inf)
                order = scores.argsort(descending=True, stable=True)
                token_mask[i] = False
                token_mask[i, order[:count]] = True
        active = token_mask & admitted[:, None]
        # Rejected tokens never exponentiate extreme ratios. No IS clipping.
        ratio = torch.where(active, current.detach() - behavior, 0.0).exp()
        if not torch.isfinite(ratio).all():
            raise FloatingPointError("importance ratio overflow on an admitted token")
        weights = torch.where(active, ratio * advantages[:, None], 0.0)

    loss = -((weights * current).sum(-1) / lengths).mean()
    return loss, {
        "advantages": advantages.detach(),
        "sequence_kl": sequence_kl.detach(),
        "admitted": admitted.detach(),
        "acceptance_rate": admitted.float().mean().detach(),
        "policy_tokens": lengths.detach(),
        "retained_tokens": active.sum(-1).detach(),
    }
