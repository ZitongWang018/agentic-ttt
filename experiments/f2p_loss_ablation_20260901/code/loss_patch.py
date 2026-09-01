from __future__ import annotations

import json
import math
import os
import statistics
from typing import Any, Dict, Iterable, List

import torch
from torch.optim import AdamW

from agents.parametric.feedback_to_policy_ttt_agent import (
    FeedbackToPolicyTTTAgent,
)
from utils import atomic_write


VALID_LOSS_MODES = {"original", "no_w", "normalized_logp_l2"}


def normalized_logp_l2_loss(
    token_logps: torch.Tensor,
    *,
    alpha: float,
    beta: float,
    nll_scale: float,
    l2_scale: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized total loss and its unnormalized component values."""
    nll = -token_logps.mean()
    l2 = torch.linalg.vector_norm(token_logps, ord=2)
    loss = alpha * nll / max(float(nll_scale), eps)
    loss = loss + beta * l2 / max(float(l2_scale), eps)
    return loss, nll, l2


def robust_positive_scale(values: Iterable[float], eps: float = 1e-8) -> float:
    finite = [abs(float(value)) for value in values if math.isfinite(float(value))]
    if not finite:
        return 1.0
    return max(float(statistics.median(finite)), eps)


def install_loss_patch(*, loss_mode: str, alpha: float, beta: float) -> None:
    """Patch only the in-process F2P class used by the isolated runner."""
    if loss_mode not in VALID_LOSS_MODES:
        raise ValueError(f"Unsupported loss mode: {loss_mode}")
    if alpha < 0 or beta < 0:
        raise ValueError("alpha and beta must be non-negative")

    original_init = FeedbackToPolicyTTTAgent.__init__
    original_policy_update = FeedbackToPolicyTTTAgent._policy_update
    original_save_memory = FeedbackToPolicyTTTAgent.save_memory
    original_load_memory = FeedbackToPolicyTTTAgent.load_memory

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.ablation_loss_mode = loss_mode
        self.ablation_alpha = float(alpha)
        self.ablation_beta = float(beta)
        self.ablation_nll_scale = None
        self.ablation_l2_scale = None
        print(
            "[F2PLossAblation] "
            f"mode={loss_mode} alpha={alpha} beta={beta}",
            flush=True,
        )

    def score_action_tokens(
        self,
        action: str,
        *,
        base_prompt: str | None,
        requires_grad: bool,
        outcome: str = "",
    ) -> torch.Tensor:
        prompt_text = self._chat_prompt(outcome, base_prompt=base_prompt)
        tokenizer = self.tlm.tokenizer
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(prompt_text + action, add_special_tokens=False)["input_ids"]
        action_start = len(prompt_ids)
        if len(full_ids) <= action_start:
            return torch.zeros(
                1, device=self.tlm.device, requires_grad=requires_grad
            )

        keep = min(len(full_ids), self.f2p_max_score_len)
        start = max(0, len(full_ids) - keep)
        input_ids = torch.tensor(
            full_ids[start:], dtype=torch.long, device=self.tlm.device
        ).unsqueeze(0)
        context_shift = max(0, action_start - start - 1)
        model = self.tlm.model
        if not requires_grad:
            model.eval()
        with torch.set_grad_enabled(requires_grad):
            # Qwen3 can apply the LM head only to the requested suffix.  The
            # extra position predicts the first action token; the final
            # position is discarded because it has no next-token label.
            first = min(context_shift, input_ids.shape[1] - 1)
            action_token_count = input_ids.shape[1] - 1 - first
            if action_token_count <= 0:
                return torch.zeros(
                    1, device=self.tlm.device, requires_grad=requires_grad
                )
            output = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                logits_to_keep=action_token_count + 1,
            )
            logits = output.logits[:, :-1, :].float()
            action_labels = input_ids[:, -action_token_count:]
            return (
                torch.log_softmax(logits, dim=-1)
                .gather(-1, action_labels.unsqueeze(-1))
                .squeeze(0)
                .squeeze(-1)
            )

    def optimized_score_action(
        self,
        outcome: str,
        action: str,
        requires_grad: bool = False,
        base_prompt: str | None = None,
    ) -> float | torch.Tensor:
        token_logps = score_action_tokens(
            self,
            action,
            base_prompt=base_prompt,
            requires_grad=requires_grad,
            outcome=outcome,
        )
        return token_logps.mean()

    def calibrate_scales(self, batch: List[Dict[str, Any]]) -> None:
        if self.ablation_nll_scale is not None and self.ablation_l2_scale is not None:
            return
        nll_values: List[float] = []
        l2_values: List[float] = []
        for item in batch:
            token_logps = score_action_tokens(
                self,
                item["action"],
                base_prompt=item["prompt"],
                requires_grad=False,
            )
            nll_values.append(float((-token_logps.mean()).detach().cpu()))
            l2_values.append(
                float(torch.linalg.vector_norm(token_logps, ord=2).detach().cpu())
            )
        self.ablation_nll_scale = robust_positive_scale(nll_values)
        self.ablation_l2_scale = robust_positive_scale(l2_values)

    def patched_policy_update(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if self.ablation_loss_mode == "original":
            result = original_policy_update(self, batch)
            result.update(
                {
                    "loss_mode": "original",
                    "alpha": None,
                    "beta": None,
                    "nll_scale": None,
                    "l2_scale": None,
                }
            )
            return result
        if not batch:
            return {"updated": False, "loss_mode": self.ablation_loss_mode}

        peft = self.tlm._ensure_peft()
        peft.train()
        optimizer = AdamW(peft.parameters(), lr=self.lr)
        optimizer.zero_grad(set_to_none=True)

        if self.ablation_loss_mode == "normalized_logp_l2":
            calibrate_scales(self, batch)
            # Calibration runs in eval mode; restore training mode before the
            # differentiable passes.
            peft.train()

        losses: List[float] = []
        nll_values: List[float] = []
        l2_values: List[float] = []
        current_scores: List[float] = []
        for item in batch:
            token_logps = score_action_tokens(
                self,
                item["action"],
                base_prompt=item["prompt"],
                requires_grad=True,
            )
            current_scores.append(float(token_logps.mean().detach().cpu()))
            if self.ablation_loss_mode == "no_w":
                nll = -token_logps.mean()
                l2 = torch.linalg.vector_norm(token_logps, ord=2)
                loss = nll
            else:
                loss, nll, l2 = normalized_logp_l2_loss(
                    token_logps,
                    alpha=self.ablation_alpha,
                    beta=self.ablation_beta,
                    nll_scale=self.ablation_nll_scale,
                    l2_scale=self.ablation_l2_scale,
                )
            losses.append(float(loss.detach().cpu()))
            nll_values.append(float(nll.detach().cpu()))
            l2_values.append(float(l2.detach().cpu()))
            (loss / len(batch)).backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(peft.parameters(), max_norm=1.0)
        optimizer.step()
        peft.eval()
        self.steps_trained_total += 1
        return {
            "updated": True,
            "loss_mode": self.ablation_loss_mode,
            "loss": sum(losses) / len(losses),
            "mean_nll": sum(nll_values) / len(nll_values),
            "mean_logp_l2": sum(l2_values) / len(l2_values),
            "grad_norm_before_clip": float(grad_norm.detach().cpu()),
            "current_action_scores": current_scores,
            "batch_size": len(batch),
            "alpha": self.ablation_alpha if self.ablation_loss_mode == "normalized_logp_l2" else None,
            "beta": self.ablation_beta if self.ablation_loss_mode == "normalized_logp_l2" else None,
            "nll_scale": self.ablation_nll_scale,
            "l2_scale": self.ablation_l2_scale,
        }

    def patched_save_memory(self, full_memory_dir: str) -> None:
        original_save_memory(self, full_memory_dir)
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["f2p_loss_ablation"] = {
            "loss_mode": self.ablation_loss_mode,
            "alpha": self.ablation_alpha,
            "beta": self.ablation_beta,
            "nll_scale": self.ablation_nll_scale,
            "l2_scale": self.ablation_l2_scale,
        }
        atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))

    def patched_load_memory(self, full_memory_dir: str) -> None:
        original_load_memory(self, full_memory_dir)
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        state = data.get("f2p_loss_ablation", {}) or {}
        saved_mode = state.get("loss_mode")
        if saved_mode and saved_mode != self.ablation_loss_mode:
            raise ValueError(
                f"Checkpoint loss mode {saved_mode!r} does not match "
                f"requested mode {self.ablation_loss_mode!r}"
            )
        self.ablation_nll_scale = state.get("nll_scale")
        self.ablation_l2_scale = state.get("l2_scale")

    FeedbackToPolicyTTTAgent.__init__ = patched_init
    FeedbackToPolicyTTTAgent._score_action = optimized_score_action
    FeedbackToPolicyTTTAgent._policy_update = patched_policy_update
    FeedbackToPolicyTTTAgent.save_memory = patched_save_memory
    FeedbackToPolicyTTTAgent.load_memory = patched_load_memory
