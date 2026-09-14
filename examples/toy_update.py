"""CPU smoke example: one fresh batch, one optimizer step, no model download."""

import torch

from flashreinforce import flashreinforce_loss

torch.manual_seed(7)
logits = torch.nn.Parameter(torch.randn(4, 6, 8))
optimizer = torch.optim.AdamW([logits], lr=1e-3)
with torch.no_grad():
    behavior_distribution = logits.softmax(-1)
    actions = torch.multinomial(behavior_distribution.flatten(0, 1), 1).reshape(4, 6)
    behavior_logps = behavior_distribution.log().gather(-1, actions[..., None]).squeeze(-1)
mask = torch.arange(6)[None, :] < torch.tensor([2, 6, 3, 5])[:, None]
current_logps = logits.log_softmax(-1).gather(-1, actions[..., None]).squeeze(-1)
loss, stats = flashreinforce_loss(current_logps, behavior_logps, torch.tensor([1., 0., 1., 0.]), mask)
optimizer.zero_grad()
loss.backward()
optimizer.step()
print(f"One update: loss={loss.item():.6f}, acceptance={stats['acceptance_rate'].item():.1%}")
