from __future__ import annotations

import math

import torch

from loss_patch import normalized_logp_l2_loss, robust_positive_scale


def main() -> None:
    token_logps = torch.tensor([-2.0, -1.0], requires_grad=True)
    loss, nll, l2 = normalized_logp_l2_loss(
        token_logps,
        alpha=1.0,
        beta=0.5,
        nll_scale=1.5,
        l2_scale=math.sqrt(5.0),
    )
    assert torch.allclose(nll, torch.tensor(1.5))
    assert torch.allclose(l2, torch.tensor(math.sqrt(5.0)), atol=1e-6)
    assert torch.allclose(loss, torch.tensor(1.5), atol=1e-6)
    loss.backward()
    # Gradient descent should move both negative log-probabilities toward zero.
    assert torch.all(token_logps.grad < 0)

    no_w_logps = torch.tensor([-2.0, -1.0], requires_grad=True)
    (-no_w_logps.mean()).backward()
    assert torch.all(no_w_logps.grad < 0)

    assert robust_positive_scale([1.0, 100.0, 2.0]) == 2.0
    assert robust_positive_scale([float("nan")]) == 1.0
    print("loss math smoke test passed")


if __name__ == "__main__":
    main()
