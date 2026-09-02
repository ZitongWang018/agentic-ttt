from __future__ import annotations

import json
import math
import os
import random
import time
from typing import Any, Dict, Iterable, List, Sequence

import torch
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from torch.optim import AdamW

from agents.parametric.feedback_to_policy_ttt_agent import FeedbackToPolicyTTTAgent
from utils import atomic_write


TASK_ADAPTER = "default"
NOVELTY_ADAPTER = "novelty"
VALID_NOVELTY_MODES = {
    "off",
    "learned_fixed_cosine",
    "learned_fixed_constant",
    "learned_unconstrained_cosine",
    "random_fixed_cosine",
}
VALID_NEGATIVE_SELECTIONS = {
    "uniform",
    "hard_task_logprob",
}
VALID_STRENGTH_MODES = {
    "absolute",
    "relative_task_norm",
}


def cosine_apply_weight(step: int, *, initial: float, decay_end: int) -> float:
    if initial < 0:
        raise ValueError("initial novelty weight must be non-negative")
    if decay_end <= 0 or step >= decay_end:
        return 0.0
    progress = max(0.0, min(float(step) / float(decay_end), 1.0))
    return float(initial) * 0.5 * (1.0 + math.cos(math.pi * progress))


def deterministic_window_sample(
    values: Sequence[str], *, sample_size: int, seed: int, step: int
) -> tuple[List[int], List[str]]:
    if sample_size <= 0 or not values:
        return [], []
    count = min(int(sample_size), len(values))
    rng = random.Random(int(seed) + 1_000_003 * int(step))
    # Sampling determines the set; FIFO ordering prevents dropout/microbatch
    # order from becoming an unintended experimental variable.
    indices = sorted(rng.sample(range(len(values)), count))
    return indices, [values[index] for index in indices]


def select_hard_negatives(
    values: Sequence[str], scores: Sequence[float], *, sample_size: int
) -> tuple[List[int], List[str]]:
    """Select the highest-current-logprob FIFO occurrences deterministically."""
    if len(values) != len(scores):
        raise ValueError("hard-negative values and scores must have equal length")
    if sample_size <= 0 or not values:
        return [], []
    count = min(int(sample_size), len(values))
    # Python's sort is stable, so FIFO index is the deterministic tie-breaker.
    hard_set = sorted(
        range(len(values)), key=lambda index: -float(scores[index])
    )[:count]
    indices = sorted(hard_set)
    return indices, [values[index] for index in indices]


def resolve_apply_weight(
    scheduled_strength: float,
    *,
    strength_mode: str,
    task_norm: float,
    novelty_norm: float,
) -> float:
    if strength_mode == "absolute" or scheduled_strength <= 0.0:
        return float(scheduled_strength)
    if strength_mode != "relative_task_norm":
        raise ValueError(f"Unsupported strength mode {strength_mode!r}")
    if (
        not math.isfinite(task_norm)
        or not math.isfinite(novelty_norm)
        or task_norm <= 1e-12
        or novelty_norm <= 1e-12
    ):
        return 0.0
    return float(scheduled_strength) * float(task_norm) / float(novelty_norm)


def score_action_means_batched(
    agent,
    prompt: str,
    actions: Sequence[str],
    *,
    requires_grad: bool,
) -> torch.Tensor:
    """Score several action suffixes while sharing one padded model forward."""
    if not actions:
        return torch.empty(0, dtype=torch.float32, device=agent.tlm.device)

    tokenizer = agent.tlm.tokenizer
    prompt_text = agent._chat_prompt("", base_prompt=prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    sequences: List[List[int]] = []
    action_token_counts: List[int] = []
    for action in actions:
        full_ids = tokenizer(
            prompt_text + str(action), add_special_tokens=False
        )["input_ids"]
        action_start = len(prompt_ids)
        if len(full_ids) <= action_start:
            sequences.append(full_ids[-1:] or [tokenizer.eos_token_id])
            action_token_counts.append(0)
            continue
        keep = min(len(full_ids), agent.f2p_max_score_len)
        start = max(0, len(full_ids) - keep)
        sequence = full_ids[start:]
        context_shift = max(0, action_start - start - 1)
        first = min(context_shift, len(sequence) - 1)
        action_token_count = len(sequence) - 1 - first
        sequences.append(sequence)
        action_token_counts.append(max(0, action_token_count))

    max_length = max(len(sequence) for sequence in sequences)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    input_ids = torch.full(
        (len(sequences), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=agent.tlm.device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, -length:] = torch.tensor(
            sequence, dtype=torch.long, device=agent.tlm.device
        )
        attention_mask[row, -length:] = 1

    max_action_tokens = max(action_token_counts)
    if max_action_tokens <= 0:
        return torch.zeros(
            len(actions),
            dtype=torch.float32,
            device=agent.tlm.device,
            requires_grad=requires_grad,
        )
    model = agent.tlm.model
    if not requires_grad:
        model.eval()
    with torch.set_grad_enabled(requires_grad):
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            logits_to_keep=max_action_tokens + 1,
        )
        logits = output.logits.float()
        means: List[torch.Tensor] = []
        for row, action_token_count in enumerate(action_token_counts):
            if action_token_count <= 0:
                means.append(logits[row].sum() * 0.0)
                continue
            action_logits = logits[
                row, -(action_token_count + 1) : -1, :
            ]
            action_labels = input_ids[row, -action_token_count:]
            token_logps = (
                torch.log_softmax(action_logits, dim=-1)
                .gather(-1, action_labels.unsqueeze(-1))
                .squeeze(-1)
            )
            means.append(token_logps.mean())
        return torch.stack(means)


def factorized_delta_sq_norm(
    lora_a: torch.Tensor, lora_b: torch.Tensor, *, scaling: float
) -> torch.Tensor:
    """Exact ||scaling * B @ A||_F^2 without materializing B @ A."""
    a = lora_a.float()
    b = lora_b.float()
    gram_a = a @ a.transpose(0, 1)
    gram_b = b.transpose(0, 1) @ b
    return (gram_a * gram_b).sum() * float(scaling) ** 2


def _lora_layers(model) -> Iterable[LoraLayer]:
    for module in model.modules():
        if isinstance(module, LoraLayer):
            yield module


def adapter_effective_norm(model, adapter_name: str) -> float:
    total = None
    for layer in _lora_layers(model):
        if adapter_name not in layer.lora_A or adapter_name not in layer.lora_B:
            continue
        rank = int(layer.r[adapter_name])
        alpha = float(layer.lora_alpha[adapter_name])
        if layer.use_rslora.get(adapter_name, False):
            scaling = alpha / math.sqrt(rank)
        else:
            scaling = alpha / rank
        value = factorized_delta_sq_norm(
            layer.lora_A[adapter_name].weight,
            layer.lora_B[adapter_name].weight,
            scaling=scaling,
        )
        total = value if total is None else total + value
    if total is None:
        return 0.0
    return float(torch.sqrt(torch.clamp(total, min=0.0)).detach().cpu())


@torch.no_grad()
def project_adapter_effective_norm(
    model, adapter_name: str, target_norm: float, eps: float = 1e-12
) -> tuple[float, float]:
    before = adapter_effective_norm(model, adapter_name)
    if not math.isfinite(before) or before <= eps:
        raise ValueError(
            f"Cannot project adapter {adapter_name!r} from effective norm {before}"
        )
    scale = float(target_norm) / before
    for layer in _lora_layers(model):
        if adapter_name in layer.lora_B:
            layer.lora_B[adapter_name].weight.mul_(scale)
    after = adapter_effective_norm(model, adapter_name)
    return before, after


def _set_adapter_scale(model, adapter_name: str, scale: float) -> None:
    for layer in _lora_layers(model):
        layer.set_scale(adapter_name, float(scale))


def _set_active_adapters(
    peft,
    adapter_names: Sequence[str],
    *,
    trainable_adapter: str | None,
    novelty_scale: float,
) -> None:
    active = list(adapter_names)
    peft.base_model.set_adapter(
        active[0] if len(active) == 1 else active,
        inference_mode=True,
    )
    for adapter_name in peft.peft_config:
        peft.set_requires_grad(adapter_name, False)
    if trainable_adapter is not None:
        peft.set_requires_grad(trainable_adapter, True)
    _set_adapter_scale(peft, TASK_ADAPTER, 1.0)
    if NOVELTY_ADAPTER in peft.peft_config:
        _set_adapter_scale(peft, NOVELTY_ADAPTER, novelty_scale)


def _adapter_parameters(peft, adapter_name: str) -> List[torch.nn.Parameter]:
    marker = f".{adapter_name}."
    return [
        parameter
        for name, parameter in peft.named_parameters()
        if marker in name and parameter.requires_grad
    ]


@torch.no_grad()
def _initialize_random_b(model, *, seed: int) -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    for layer in _lora_layers(model):
        if NOVELTY_ADAPTER not in layer.lora_B:
            continue
        weight = layer.lora_B[NOVELTY_ADAPTER].weight
        random_weight = torch.randn(
            tuple(weight.shape), generator=generator, dtype=torch.float32
        )
        weight.copy_(random_weight.to(device=weight.device, dtype=weight.dtype))


def install_dual_lora_novelty_patch(
    *,
    novelty_mode: str,
    novelty_rank: int,
    novelty_alpha: int,
    novelty_lr: float,
    novelty_update_frequency: int,
    novelty_window_size: int,
    novelty_batch_size: int,
    novelty_lambda0: float,
    novelty_decay_end: int,
    novelty_seed: int,
    novelty_negative_selection: str = "uniform",
    novelty_strength_mode: str = "absolute",
    novelty_score_batch_size: int = 5,
    novelty_train_microbatch_size: int = 1,
) -> None:
    if novelty_mode not in VALID_NOVELTY_MODES:
        raise ValueError(f"Unsupported novelty mode: {novelty_mode}")
    if novelty_rank <= 0 or novelty_alpha <= 0:
        raise ValueError("novelty rank and alpha must be positive")
    if novelty_lr <= 0:
        raise ValueError("novelty learning rate must be positive")
    if novelty_update_frequency <= 0:
        raise ValueError("novelty update frequency must be positive")
    if novelty_window_size <= 0 or novelty_batch_size <= 0:
        raise ValueError("novelty window and batch size must be positive")
    if novelty_batch_size > novelty_window_size:
        raise ValueError("novelty batch size cannot exceed its history window")
    if novelty_negative_selection not in VALID_NEGATIVE_SELECTIONS:
        raise ValueError(
            f"Unsupported negative selection {novelty_negative_selection!r}"
        )
    if novelty_strength_mode not in VALID_STRENGTH_MODES:
        raise ValueError(f"Unsupported strength mode {novelty_strength_mode!r}")
    if novelty_score_batch_size <= 0 or novelty_train_microbatch_size <= 0:
        raise ValueError("novelty score batch sizes must be positive")

    original_init = FeedbackToPolicyTTTAgent.__init__
    original_act = FeedbackToPolicyTTTAgent._act
    original_observe_transition = FeedbackToPolicyTTTAgent.observe_transition
    original_policy_update = FeedbackToPolicyTTTAgent._policy_update
    original_save_memory = FeedbackToPolicyTTTAgent.save_memory
    original_load_memory = FeedbackToPolicyTTTAgent.load_memory

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.novelty_mode = novelty_mode
        self.novelty_rank = int(novelty_rank)
        self.novelty_alpha = int(novelty_alpha)
        self.novelty_lr = float(novelty_lr)
        self.novelty_update_frequency = int(novelty_update_frequency)
        self.novelty_window_size = int(novelty_window_size)
        self.novelty_batch_size = int(novelty_batch_size)
        self.novelty_lambda0 = float(novelty_lambda0)
        self.novelty_decay_end = int(novelty_decay_end)
        self.novelty_seed = int(novelty_seed)
        self.novelty_negative_selection = novelty_negative_selection
        self.novelty_strength_mode = novelty_strength_mode
        self.novelty_score_batch_size = int(novelty_score_batch_size)
        self.novelty_train_microbatch_size = int(novelty_train_microbatch_size)
        self.novelty_action_history: List[str] = []
        self.novelty_steps_seen = 0
        self.novelty_updates_total = 0
        self.novelty_last_update_step = -1
        self.novelty_target_norm: float | None = None
        self.novelty_last_trace: Dict[str, Any] = {}
        self.novelty_random_initialized = False
        print(
            "[DualLoRANovelty] "
            f"mode={self.novelty_mode} rank={self.novelty_rank} "
            f"alpha={self.novelty_alpha} lr={self.novelty_lr} "
            f"window={self.novelty_window_size} batch={self.novelty_batch_size} "
            f"frequency={self.novelty_update_frequency} "
            f"lambda0={self.novelty_lambda0} decay_end={self.novelty_decay_end} "
            f"selection={self.novelty_negative_selection} "
            f"strength={self.novelty_strength_mode} "
            f"score_batch={self.novelty_score_batch_size} "
            f"train_microbatch={self.novelty_train_microbatch_size}",
            flush=True,
        )

    def ensure_dual_adapters(self):
        peft = self.tlm._ensure_peft()
        if self.novelty_mode == "off":
            return peft
        if NOVELTY_ADAPTER not in peft.peft_config:
            config = LoraConfig(
                r=self.novelty_rank,
                lora_alpha=self.novelty_alpha,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                bias="none",
                task_type="CAUSAL_LM",
            )
            peft.add_adapter(NOVELTY_ADAPTER, config)
        return peft

    def novelty_scheduled_strength(self, step: int | None = None) -> float:
        if self.novelty_mode == "off":
            return 0.0
        if self.novelty_mode == "learned_fixed_constant":
            return self.novelty_lambda0
        current_step = self.novelty_steps_seen if step is None else int(step)
        return cosine_apply_weight(
            current_step,
            initial=self.novelty_lambda0,
            decay_end=self.novelty_decay_end,
        )

    def novelty_apply_weight(self, step: int | None = None) -> float:
        scheduled = novelty_scheduled_strength(self, step)
        if self.novelty_strength_mode == "absolute":
            return scheduled
        peft = self.tlm._ensure_peft()
        if NOVELTY_ADAPTER not in peft.peft_config:
            return 0.0
        task_norm = adapter_effective_norm(peft, TASK_ADAPTER)
        novelty_norm = adapter_effective_norm(peft, NOVELTY_ADAPTER)
        # The scheduled value is the desired applied novelty/task norm ratio.
        return resolve_apply_weight(
            scheduled,
            strength_mode=self.novelty_strength_mode,
            task_norm=task_norm,
            novelty_norm=novelty_norm,
        )

    def configure_generation(self) -> None:
        peft = ensure_dual_adapters(self)
        peft.eval()
        if self.novelty_mode == "off":
            _set_active_adapters(
                peft,
                [TASK_ADAPTER],
                trainable_adapter=None,
                novelty_scale=0.0,
            )
            return
        _set_active_adapters(
            peft,
            [TASK_ADAPTER, NOVELTY_ADAPTER],
            trainable_adapter=None,
            novelty_scale=novelty_apply_weight(self),
        )

    def score_negative_actions(
        self, prompt: str, actions: Sequence[str]
    ) -> List[float]:
        scores: List[float] = []
        with torch.no_grad():
            for start in range(0, len(actions), self.novelty_score_batch_size):
                chunk = actions[start : start + self.novelty_score_batch_size]
                values = score_action_means_batched(
                    self, prompt, chunk, requires_grad=False
                )
                scores.extend(
                    float(value) for value in values.detach().float().cpu().tolist()
                )
        return scores

    def train_negative_actions(
        self, prompt: str, actions: Sequence[str]
    ) -> List[float]:
        losses: List[float] = []
        for start in range(
            0, len(actions), self.novelty_train_microbatch_size
        ):
            chunk = actions[start : start + self.novelty_train_microbatch_size]
            scores = score_action_means_batched(
                self, prompt, chunk, requires_grad=True
            )
            losses.extend(
                float(value) for value in scores.detach().float().cpu().tolist()
            )
            (scores.sum() / len(actions)).backward()
        return losses

    def maybe_update_novelty(self, prompt: str) -> Dict[str, Any]:
        scheduled_strength = novelty_scheduled_strength(self)
        current_lambda = novelty_apply_weight(self)
        base_trace: Dict[str, Any] = {
            "mode": self.novelty_mode,
            "step_before_action": self.novelty_steps_seen,
            "apply_weight": current_lambda,
            "scheduled_strength": scheduled_strength,
            "strength_mode": self.novelty_strength_mode,
            "negative_selection": self.novelty_negative_selection,
            "window_size": len(self.novelty_action_history),
            "updated": False,
        }
        if self.novelty_mode == "off":
            return base_trace
        if not self.novelty_action_history:
            base_trace["skip_reason"] = "empty_history"
            return base_trace
        if self.novelty_steps_seen % self.novelty_update_frequency != 0:
            base_trace["skip_reason"] = "not_update_boundary"
            return base_trace
        if self.novelty_last_update_step == self.novelty_steps_seen:
            base_trace["skip_reason"] = "already_updated"
            return base_trace
        if (
            scheduled_strength <= 0.0
            and self.novelty_mode != "learned_fixed_constant"
        ):
            base_trace["skip_reason"] = "zero_apply_weight"
            return base_trace

        update_started = time.perf_counter()
        peft = ensure_dual_adapters(self)
        task_norm = adapter_effective_norm(peft, TASK_ADAPTER)
        if self.novelty_target_norm is None:
            if not math.isfinite(task_norm) or task_norm <= 1e-12:
                base_trace["skip_reason"] = "task_norm_not_ready"
                base_trace["task_effective_norm"] = task_norm
                return base_trace
            self.novelty_target_norm = task_norm

        selection_started = time.perf_counter()
        candidate_actions = list(self.novelty_action_history)
        _set_active_adapters(
            peft,
            [TASK_ADAPTER],
            trainable_adapter=None,
            novelty_scale=0.0,
        )
        peft.eval()
        if self.novelty_negative_selection == "hard_task_logprob":
            candidate_task_scores = score_negative_actions(
                self, prompt, candidate_actions
            )
            candidate_scored_actions = candidate_actions
            indices, negatives = select_hard_negatives(
                candidate_actions,
                candidate_task_scores,
                sample_size=self.novelty_batch_size,
            )
            task_only_scores = [candidate_task_scores[index] for index in indices]
        else:
            indices, negatives = deterministic_window_sample(
                candidate_actions,
                sample_size=self.novelty_batch_size,
                seed=self.novelty_seed,
                step=self.novelty_steps_seen,
            )
            task_only_scores = score_negative_actions(self, prompt, negatives)
            candidate_task_scores = task_only_scores
            candidate_scored_actions = negatives
        selection_seconds = time.perf_counter() - selection_started

        pre_apply_weight = novelty_apply_weight(self)
        _set_active_adapters(
            peft,
            [TASK_ADAPTER, NOVELTY_ADAPTER],
            trainable_adapter=None,
            novelty_scale=pre_apply_weight,
        )
        pre_applied_scores = score_negative_actions(self, prompt, negatives)
        _set_active_adapters(
            peft,
            [TASK_ADAPTER, NOVELTY_ADAPTER],
            trainable_adapter=None,
            novelty_scale=1.0,
        )
        pre_scores = score_negative_actions(self, prompt, negatives)

        projection_before = None
        projection_after = None
        loss_value = None
        grad_norm_value = None

        if self.novelty_mode == "random_fixed_cosine":
            if not self.novelty_random_initialized:
                _initialize_random_b(
                    peft,
                    seed=self.novelty_seed + 17_171,
                )
                projection_before, projection_after = project_adapter_effective_norm(
                    peft, NOVELTY_ADAPTER, self.novelty_target_norm
                )
                self.novelty_random_initialized = True
        else:
            _set_active_adapters(
                peft,
                [TASK_ADAPTER, NOVELTY_ADAPTER],
                trainable_adapter=NOVELTY_ADAPTER,
                novelty_scale=1.0,
            )
            peft.train()
            for layer in _lora_layers(peft):
                if TASK_ADAPTER in layer.lora_dropout:
                    layer.lora_dropout[TASK_ADAPTER].eval()
                if NOVELTY_ADAPTER in layer.lora_dropout:
                    layer.lora_dropout[NOVELTY_ADAPTER].train()
            parameters = _adapter_parameters(peft, NOVELTY_ADAPTER)
            if not parameters:
                raise RuntimeError("No trainable novelty adapter parameters found")
            optimizer = AdamW(parameters, lr=self.novelty_lr)
            optimizer.zero_grad(set_to_none=True)
            training_started = time.perf_counter()
            losses = train_negative_actions(self, prompt, negatives)
            grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            training_seconds = time.perf_counter() - training_started
            loss_value = sum(losses) / len(losses)
            grad_norm_value = float(grad_norm.detach().cpu())

            should_project = (
                self.novelty_mode != "learned_unconstrained_cosine"
                or self.novelty_updates_total == 0
            )
            if should_project:
                projection_before, projection_after = project_adapter_effective_norm(
                    peft, NOVELTY_ADAPTER, self.novelty_target_norm
                )

        peft.eval()
        post_started = time.perf_counter()
        _set_active_adapters(
            peft,
            [TASK_ADAPTER, NOVELTY_ADAPTER],
            trainable_adapter=None,
            novelty_scale=1.0,
        )
        post_scores = score_negative_actions(self, prompt, negatives)
        current_lambda = novelty_apply_weight(self)
        _set_active_adapters(
            peft,
            [TASK_ADAPTER, NOVELTY_ADAPTER],
            trainable_adapter=None,
            novelty_scale=current_lambda,
        )
        post_applied_scores = score_negative_actions(self, prompt, negatives)
        post_seconds = time.perf_counter() - post_started
        self.novelty_last_update_step = self.novelty_steps_seen
        self.novelty_updates_total += 1
        novelty_norm = adapter_effective_norm(peft, NOVELTY_ADAPTER)
        relative_applied_strength = (
            current_lambda * novelty_norm / task_norm
            if task_norm > 1e-12
            else None
        )
        mean_task_only = sum(task_only_scores) / len(task_only_scores)
        mean_pre_applied = sum(pre_applied_scores) / len(pre_applied_scores)
        mean_post_applied = sum(post_applied_scores) / len(post_applied_scores)
        mean_pre_full = sum(pre_scores) / len(pre_scores)
        mean_post_full = sum(post_scores) / len(post_scores)
        trace = {
            **base_trace,
            "updated": True,
            "apply_weight_before_update": pre_apply_weight,
            "apply_weight": current_lambda,
            "scheduled_strength": scheduled_strength,
            "relative_applied_strength": relative_applied_strength,
            "sample_indices": indices,
            "sampled_actions": negatives,
            "candidate_count": len(candidate_actions),
            "candidate_actions": candidate_actions,
            "candidate_scored_actions": candidate_scored_actions,
            "candidate_task_only_scores": candidate_task_scores,
            "selected_task_only_scores": task_only_scores,
            "negative_mean_logprob_before": mean_pre_full,
            "negative_mean_logprob_after": mean_post_full,
            "negative_mean_logprob_change": mean_post_full - mean_pre_full,
            "negative_scores_before": pre_scores,
            "negative_scores_after": post_scores,
            "counterfactual_task_only_scores": task_only_scores,
            "counterfactual_applied_scores_before": pre_applied_scores,
            "counterfactual_applied_scores_after": post_applied_scores,
            "counterfactual_full_scores_before": pre_scores,
            "counterfactual_full_scores_after": post_scores,
            "counterfactual_applied_minus_task_mean_after": (
                mean_post_applied - mean_task_only
            ),
            "counterfactual_full_minus_task_mean_after": (
                mean_post_full - mean_task_only
            ),
            "counterfactual_applied_update_change": (
                mean_post_applied - mean_pre_applied
            ),
            "loss": loss_value,
            "grad_norm_before_clip": grad_norm_value,
            "task_effective_norm": task_norm,
            "target_effective_norm": self.novelty_target_norm,
            "novelty_effective_norm": novelty_norm,
            "effective_applied_norm": current_lambda * novelty_norm,
            "projection_norm_before": projection_before,
            "projection_norm_after": projection_after,
            "updates_total": self.novelty_updates_total,
            "loss_definition": "mean_token_logprob(historical_action | current_context)",
            "timing_seconds": {
                "selection_and_task_only_scoring": selection_seconds,
                "training": training_seconds
                if self.novelty_mode != "random_fixed_cosine"
                else 0.0,
                "post_counterfactual_scoring": post_seconds,
                "total_update": time.perf_counter() - update_started,
            },
        }
        return trace

    def append_novelty_trace(self, trace: Dict[str, Any]) -> None:
        if not self.f2p_log_path:
            return
        path = os.path.join(
            os.path.dirname(self.f2p_log_path), "novelty_intermediates.jsonl"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(trace, ensure_ascii=False) + "\n")

    def patched_act(self, obs: Dict[str, Any]):
        if self.novelty_mode == "off":
            configure_generation(self)
            self.novelty_last_trace = {
                "mode": "off",
                "step_before_action": self.novelty_steps_seen,
                "apply_weight": 0.0,
                "updated": False,
            }
            return original_act(self, obs)

        obs_text = obs.get("text", "")
        memory_text = "\n\n".join(self.short_term_memory)
        self._last_action_prompt = (
            "My Current Observation:\n" + obs_text
            + (
                "\n\nVerified recent feedback:\n" + memory_text
                if memory_text
                else ""
            )
            + "\n\nChoose one valid action and briefly predict its immediate environment change."
        )
        self.novelty_last_trace = maybe_update_novelty(
            self, self._last_action_prompt
        )
        configure_generation(self)
        self.novelty_last_trace["generation_apply_weight"] = novelty_apply_weight(
            self
        )

        lm_output = self.tlm.generate(
            user_prompt=self._last_action_prompt,
            system_prompt=self.f2p_system,
        )
        parsed = self.cfg.json_parser(lm_output["response"])
        if not isinstance(parsed, dict):
            parsed = {
                "action": "wait",
                "predicted_outcome": "",
                "reasoning": "parse failure",
            }
        action = parsed.get("action", "wait")
        if not isinstance(action, str) or not action.strip():
            action = "wait"
        self._last_action = action.strip()
        self._last_prediction = str(parsed.get("predicted_outcome", ""))
        if self.novelty_last_trace.get("updated"):
            peft = ensure_dual_adapters(self)
            _set_active_adapters(
                peft,
                [TASK_ADAPTER],
                trainable_adapter=None,
                novelty_scale=0.0,
            )
            task_only = score_negative_actions(
                self, self._last_action_prompt, [self._last_action]
            )[0]
            apply_weight = novelty_apply_weight(self)
            _set_active_adapters(
                peft,
                [TASK_ADAPTER, NOVELTY_ADAPTER],
                trainable_adapter=None,
                novelty_scale=apply_weight,
            )
            applied = score_negative_actions(
                self, self._last_action_prompt, [self._last_action]
            )[0]
            _set_active_adapters(
                peft,
                [TASK_ADAPTER, NOVELTY_ADAPTER],
                trainable_adapter=None,
                novelty_scale=1.0,
            )
            full = score_negative_actions(
                self, self._last_action_prompt, [self._last_action]
            )[0]
            configure_generation(self)
            self._last_action_prior = applied
            self.novelty_last_trace.update(
                {
                    "chosen_action": self._last_action,
                    "chosen_was_in_window": (
                        self._last_action in self.novelty_action_history
                    ),
                    "chosen_was_sampled": (
                        self._last_action
                        in self.novelty_last_trace.get("sampled_actions", [])
                    ),
                    "chosen_action_score_task_only": task_only,
                    "chosen_action_score_applied": applied,
                    "chosen_action_score_full": full,
                    "chosen_action_applied_minus_task": applied - task_only,
                    "chosen_action_full_minus_task": full - task_only,
                }
            )
        else:
            prior = self._score_action("", self._last_action, requires_grad=False)
            self._last_action_prior = (
                float(prior.detach().cpu())
                if torch.is_tensor(prior)
                else float(prior)
            )
        append_novelty_trace(self, self.novelty_last_trace)
        return (
            self._last_action,
            lm_output["num_input_tokens"],
            lm_output["num_output_tokens"],
            lm_output["response"],
        )

    def patched_policy_update(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        peft = ensure_dual_adapters(self)
        _set_active_adapters(
            peft,
            [TASK_ADAPTER],
            trainable_adapter=TASK_ADAPTER,
            novelty_scale=0.0,
        )
        result = original_policy_update(self, batch)
        result["task_effective_norm"] = adapter_effective_norm(peft, TASK_ADAPTER)
        result["novelty_effective_norm"] = (
            adapter_effective_norm(peft, NOVELTY_ADAPTER)
            if NOVELTY_ADAPTER in peft.peft_config
            else 0.0
        )
        result["adapter_routing"] = "task_only"
        configure_generation(self)
        return result

    def patched_observe_transition(
        self, previous_obs, action: str, next_obs, reward=None, info=None
    ):
        original_observe_transition(
            self,
            previous_obs,
            action,
            next_obs,
            reward=reward,
            info=info,
        )
        self.novelty_action_history.append(str(action))
        self.novelty_action_history = self.novelty_action_history[
            -self.novelty_window_size :
        ]
        self.novelty_steps_seen += 1
        self.f2p_last_trace = dict(self.f2p_last_trace)
        self.f2p_last_trace["novelty"] = dict(self.novelty_last_trace)
        self.f2p_last_trace["novelty"]["history_after_action"] = list(
            self.novelty_action_history
        )
        self.f2p_last_trace["novelty"]["steps_seen_after_action"] = (
            self.novelty_steps_seen
        )

    def patched_save_memory(self, full_memory_dir: str) -> None:
        configure_generation(self)
        original_save_memory(self, full_memory_dir)
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["dual_lora_novelty"] = {
            "mode": self.novelty_mode,
            "rank": self.novelty_rank,
            "alpha": self.novelty_alpha,
            "lr": self.novelty_lr,
            "update_frequency": self.novelty_update_frequency,
            "window_size": self.novelty_window_size,
            "batch_size": self.novelty_batch_size,
            "lambda0": self.novelty_lambda0,
            "decay_end": self.novelty_decay_end,
            "seed": self.novelty_seed,
            "negative_selection": self.novelty_negative_selection,
            "strength_mode": self.novelty_strength_mode,
            "score_batch_size": self.novelty_score_batch_size,
            "train_microbatch_size": self.novelty_train_microbatch_size,
            "action_history": list(self.novelty_action_history),
            "steps_seen": self.novelty_steps_seen,
            "updates_total": self.novelty_updates_total,
            "last_update_step": self.novelty_last_update_step,
            "target_norm": self.novelty_target_norm,
            "last_trace": self.novelty_last_trace,
            "random_initialized": self.novelty_random_initialized,
        }
        atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))

    def patched_load_memory(self, full_memory_dir: str) -> None:
        original_load_memory(self, full_memory_dir)
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        state = data.get("dual_lora_novelty", {}) or {}
        saved_mode = state.get("mode")
        if saved_mode and saved_mode != self.novelty_mode:
            raise ValueError(
                f"Checkpoint novelty mode {saved_mode!r} does not match "
                f"requested mode {self.novelty_mode!r}"
            )
        saved_selection = state.get("negative_selection")
        if saved_selection and saved_selection != self.novelty_negative_selection:
            raise ValueError(
                f"Checkpoint negative selection {saved_selection!r} does not match "
                f"requested selection {self.novelty_negative_selection!r}"
            )
        saved_strength = state.get("strength_mode")
        if saved_strength and saved_strength != self.novelty_strength_mode:
            raise ValueError(
                f"Checkpoint strength mode {saved_strength!r} does not match "
                f"requested mode {self.novelty_strength_mode!r}"
            )
        self.novelty_action_history = list(state.get("action_history", []) or [])[
            -self.novelty_window_size :
        ]
        self.novelty_steps_seen = int(state.get("steps_seen", 0))
        self.novelty_updates_total = int(state.get("updates_total", 0))
        self.novelty_last_update_step = int(state.get("last_update_step", -1))
        self.novelty_target_norm = state.get("target_norm")
        self.novelty_last_trace = dict(state.get("last_trace", {}) or {})
        self.novelty_random_initialized = bool(
            state.get("random_initialized", False)
        )

        peft = self.tlm._ensure_peft()
        if self.novelty_mode != "off" and NOVELTY_ADAPTER not in peft.peft_config:
            novelty_dir = os.path.join(
                full_memory_dir, self.adapter_subdir, NOVELTY_ADAPTER
            )
            if os.path.isfile(os.path.join(novelty_dir, "adapter_config.json")):
                peft.load_adapter(
                    novelty_dir,
                    adapter_name=NOVELTY_ADAPTER,
                    is_trainable=True,
                )
            else:
                ensure_dual_adapters(self)
        configure_generation(self)

    FeedbackToPolicyTTTAgent.__init__ = patched_init
    FeedbackToPolicyTTTAgent._ensure_dual_adapters = ensure_dual_adapters
    FeedbackToPolicyTTTAgent._novelty_scheduled_strength = (
        novelty_scheduled_strength
    )
    FeedbackToPolicyTTTAgent._novelty_apply_weight = novelty_apply_weight
    FeedbackToPolicyTTTAgent._configure_generation_adapters = configure_generation
    FeedbackToPolicyTTTAgent._maybe_update_novelty = maybe_update_novelty
    FeedbackToPolicyTTTAgent._act = patched_act
    FeedbackToPolicyTTTAgent._policy_update = patched_policy_update
    FeedbackToPolicyTTTAgent.observe_transition = patched_observe_transition
    FeedbackToPolicyTTTAgent.save_memory = patched_save_memory
    FeedbackToPolicyTTTAgent.load_memory = patched_load_memory
