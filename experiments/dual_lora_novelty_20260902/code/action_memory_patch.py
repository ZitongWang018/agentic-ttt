from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from transformers import LogitsProcessor, LogitsProcessorList

from agents.parametric.feedback_to_policy_ttt_agent import FeedbackToPolicyTTTAgent
from utils import atomic_write


VALID_ACTION_MEMORY_MODES = {
    "control",
    "exact_adapter",
    "semantic_adapter",
    "prompt_history",
}
PARAMETRIC_MODES = {"exact_adapter", "semantic_adapter"}


def find_json_action_span(text: str) -> tuple[int, int] | None:
    """Return the character span of the last complete JSON action value."""
    matches = list(re.finditer(r'"action"\s*:\s*"', text or ""))
    if not matches:
        return None
    start = matches[-1].end()
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return start, index
    return None


def inside_json_action_value(text: str) -> bool:
    """Whether the next generated token is inside the JSON action string."""
    matches = list(re.finditer(r'"action"\s*:\s*"', text or ""))
    if not matches:
        return False
    start = matches[-1].end()
    escaped = False
    for char in text[start:]:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            return False
    return True


def build_action_history_prompt(actions: Sequence[str]) -> str:
    if not actions:
        return ""
    lines = [
        "Recent executed actions (oldest to newest; at most 25):",
        *[f"{index}. {action}" for index, action in enumerate(actions, 1)],
        "",
        "EXPLORATION INSTRUCTION:",
        "Prefer a valid action that is materially different from every action above,",
        "even when a listed action previously changed the state or was successful.",
        "Use a listed action again only when repetition is required for immediate",
        "progress, validity, or survival and there is no useful untried alternative.",
    ]
    return "\n".join(lines)


def frozen_action_embedding(tokenizer, embedding_layer, action: str) -> torch.Tensor:
    token_ids = tokenizer(str(action), add_special_tokens=False)["input_ids"]
    if not token_ids:
        return torch.zeros(
            embedding_layer.embedding_dim, dtype=torch.float32, device="cpu"
        )
    ids = torch.tensor(token_ids, dtype=torch.long, device=embedding_layer.weight.device)
    with torch.no_grad():
        value = embedding_layer(ids).float().mean(dim=0)
        value = F.normalize(value, dim=0)
    return value.detach().cpu()


def greedy_semantic_clusters(
    embeddings: Sequence[torch.Tensor], *, threshold: float
) -> List[List[int]]:
    """FIFO-stable online cosine clustering with normalized centroids."""
    clusters: List[List[int]] = []
    centroids: List[torch.Tensor] = []
    for index, embedding in enumerate(embeddings):
        vector = F.normalize(embedding.float(), dim=0)
        if centroids:
            scores = torch.tensor(
                [float(torch.dot(vector, centroid)) for centroid in centroids]
            )
            best = int(torch.argmax(scores))
            if float(scores[best]) >= float(threshold):
                clusters[best].append(index)
                members = torch.stack(
                    [F.normalize(embeddings[item].float(), dim=0) for item in clusters[best]]
                )
                centroids[best] = F.normalize(members.mean(dim=0), dim=0)
                continue
        clusters.append([index])
        centroids.append(vector)
    return clusters


def semantic_cluster_plan(
    embeddings: Sequence[torch.Tensor], *, threshold: float
) -> tuple[List[int], List[float], List[List[int]]]:
    clusters = greedy_semantic_clusters(embeddings, threshold=threshold)
    representatives: List[int] = []
    weights: List[float] = []
    total = max(1, len(embeddings))
    for cluster in clusters:
        members = torch.stack(
            [F.normalize(embeddings[index].float(), dim=0) for index in cluster]
        )
        centroid = F.normalize(members.mean(dim=0), dim=0)
        similarities = members @ centroid
        representatives.append(cluster[int(torch.argmax(similarities))])
        weights.append(len(cluster) / total)
    return representatives, weights, clusters


@dataclass
class ActionRecord:
    action: str
    hidden: torch.Tensor
    base_logits: torch.Tensor
    labels: torch.Tensor
    source_step: int
    source: str


class LowRankActionLogitAdapter(nn.Module):
    """A small residual action head: W_up(SiLU(W_down(h)))."""

    def __init__(
        self,
        *,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        seed: int,
        protected_token_ids: Sequence[int],
    ) -> None:
        super().__init__()
        # Keep adapter construction from perturbing the policy sampling RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.down = nn.Linear(hidden_size, rank, bias=False)
            self.up = nn.Linear(rank, vocab_size, bias=False)
        nn.init.zeros_(self.up.weight)
        mask = torch.ones(vocab_size, dtype=torch.float32)
        if protected_token_ids:
            mask[list(protected_token_ids)] = 0.0
        self.register_buffer("output_mask", mask, persistent=False)
        self._initial_state = {
            key: value.detach().cpu().clone() for key, value in self.state_dict().items()
        }

    def reset_parameters_to_initial(self) -> None:
        self.load_state_dict(self._initial_state, strict=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        delta = self.up(F.silu(self.down(hidden.float())))
        return delta * self.output_mask


class _ActionSlotProcessor(LogitsProcessor):
    def __init__(self, *, agent, adapter, scale: float) -> None:
        self.agent = agent
        self.adapter = adapter
        self.scale = float(scale)
        self.prompt_length: int | None = None
        self.latest_hidden: torch.Tensor | None = None
        self.captures: List[Dict[str, Any]] = []
        self.record: ActionRecord | None = None
        self.applied_positions = 0
        self.effect_trace: Dict[str, Any] = {}

    def capture_lm_head_input(self, module, args) -> None:
        if not args:
            return
        hidden = args[0]
        if torch.is_tensor(hidden) and hidden.ndim == 3:
            self.latest_hidden = hidden[:, -1, :].detach()

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.prompt_length is None:
            self.prompt_length = int(input_ids.shape[1])
        generated = self.agent.tlm.tokenizer.decode(
            input_ids[0, self.prompt_length :],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not inside_json_action_value(generated) or self.latest_hidden is None:
            return scores
        delta = self.adapter(self.latest_hidden.to(self.adapter.down.weight.device))
        delta = delta.to(device=scores.device, dtype=scores.dtype)
        self.captures.append(
            {
                "position": int(input_ids.shape[1]),
                "hidden": self.latest_hidden[0].detach().to("cpu", dtype=torch.float16),
                "base_logits": scores[0].detach().to("cpu", dtype=torch.float16),
            }
        )
        self.applied_positions += 1
        return scores + self.scale * delta

    def finalize(self, output_ids: torch.Tensor, *, action: str, source_step: int) -> None:
        tokenizer = self.agent.tlm.tokenizer
        kept = []
        for capture in self.captures:
            position = int(capture["position"])
            if position >= int(output_ids.shape[-1]):
                continue
            label = int(output_ids[0, position]) if output_ids.ndim == 2 else int(output_ids[position])
            token_piece = tokenizer.convert_ids_to_tokens(label) or ""
            # The adapter is active while predicting the closing quote. Keep
            # structural JSON tokens unchanged and out of the negative loss.
            if any(char in token_piece for char in ('"', "{", "}")):
                continue
            capture["label"] = label
            kept.append(capture)
        if not kept:
            self.record = None
            return
        device = self.adapter.down.weight.device
        hidden = torch.stack([item["hidden"] for item in kept]).to(
            device=device, dtype=torch.float32
        )
        base_logits = torch.stack([item["base_logits"] for item in kept]).to(
            device=device, dtype=torch.float32
        )
        labels = torch.tensor(
            [item["label"] for item in kept], dtype=torch.long, device=device
        )
        with torch.no_grad():
            delta = self.scale * self.adapter(hidden)
            adapted_logits = base_logits + delta
            base_log_probs = F.log_softmax(base_logits, dim=-1)
            adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
            base_targets = base_log_probs.gather(-1, labels[:, None]).squeeze(-1)
            adapted_targets = adapted_log_probs.gather(-1, labels[:, None]).squeeze(-1)
            reference_kl = (
                base_log_probs.exp() * (base_log_probs - adapted_log_probs)
            ).sum(dim=-1)
            target_deltas = adapted_targets - base_targets
            top1_changed = base_logits.argmax(dim=-1) != adapted_logits.argmax(dim=-1)
            self.effect_trace = {
                "chosen_token_ids": labels.detach().cpu().tolist(),
                "chosen_base_token_logprobs": base_targets.detach().cpu().tolist(),
                "chosen_adapted_token_logprobs": adapted_targets.detach().cpu().tolist(),
                "chosen_target_logprob_delta_mean": float(target_deltas.mean().cpu()),
                "chosen_target_logprob_delta_min": float(target_deltas.min().cpu()),
                "chosen_target_logprob_delta_max": float(target_deltas.max().cpu()),
                "action_slot_reference_kl_mean": float(reference_kl.mean().cpu()),
                "action_slot_logit_delta_abs_mean": float(delta.abs().mean().cpu()),
                "action_slot_logit_delta_abs_max": float(delta.abs().max().cpu()),
                "action_slot_top1_changed_tokens": int(top1_changed.sum().cpu()),
            }
        self.record = ActionRecord(
            action=str(action),
            hidden=torch.stack([item["hidden"] for item in kept]),
            base_logits=torch.stack([item["base_logits"] for item in kept]),
            labels=torch.tensor([item["label"] for item in kept], dtype=torch.long),
            source_step=int(source_step),
            source="online_action_slot",
        )


def _protected_json_token_ids(tokenizer) -> List[int]:
    protected = []
    for token, token_id in tokenizer.get_vocab().items():
        if any(char in token for char in ('"', "{", "}")):
            protected.append(int(token_id))
    return sorted(set(protected))


def _weighted_record_plan(agent) -> tuple[List[ActionRecord], List[float], Dict[str, Any]]:
    records = list(agent.action_memory_records)
    if agent.action_memory_mode == "exact_adapter":
        count = max(1, len(records))
        return records, [1.0 / count] * len(records), {
            "cluster_count": None,
            "clusters": None,
            "representative_indices": list(range(len(records))),
        }

    embeddings = [
        frozen_action_embedding(
            agent.tlm.tokenizer,
            agent.tlm.model.get_input_embeddings(),
            record.action,
        )
        for record in records
    ]
    representatives, weights, clusters = semantic_cluster_plan(
        embeddings, threshold=agent.action_memory_semantic_threshold
    )
    selected = [records[index] for index in representatives]
    return selected, weights, {
        "cluster_count": len(clusters),
        "clusters": [
            {
                "indices": cluster,
                "actions": [records[index].action for index in cluster],
                "count": len(cluster),
            }
            for cluster in clusters
        ],
        "representative_indices": representatives,
    }


def _adapter_objective(
    adapter: LowRankActionLogitAdapter,
    records: Sequence[ActionRecord],
    record_weights: Sequence[float],
    *,
    reference_beta: float,
    backward: bool,
    microbatch_tokens: int,
    adapter_scale: float = 1.0,
) -> Dict[str, float]:
    device = adapter.down.weight.device
    entries = []
    for record, record_weight in zip(records, record_weights):
        token_count = max(1, int(record.labels.numel()))
        for index in range(int(record.labels.numel())):
            entries.append(
                (
                    record.hidden[index],
                    record.base_logits[index],
                    record.labels[index],
                    float(record_weight) / token_count,
                )
            )
    if not entries:
        return {"loss": 0.0, "unlikelihood": 0.0, "reference_kl": 0.0, "target_logprob": 0.0}

    totals = {"loss": 0.0, "unlikelihood": 0.0, "reference_kl": 0.0, "target_logprob": 0.0}
    for start in range(0, len(entries), max(1, int(microbatch_tokens))):
        chunk = entries[start : start + max(1, int(microbatch_tokens))]
        hidden = torch.stack([item[0] for item in chunk]).to(device=device, dtype=torch.float32)
        base_logits = torch.stack([item[1] for item in chunk]).to(device=device, dtype=torch.float32)
        labels = torch.stack([item[2] for item in chunk]).to(device=device, dtype=torch.long)
        weights = torch.tensor([item[3] for item in chunk], device=device, dtype=torch.float32)

        adapted_logits = base_logits + float(adapter_scale) * adapter(hidden)
        base_log_probs = F.log_softmax(base_logits, dim=-1)
        adapted_log_probs = F.log_softmax(adapted_logits, dim=-1)
        target_log_probs = adapted_log_probs.gather(-1, labels[:, None]).squeeze(-1)
        target_probs = target_log_probs.exp().clamp(max=1.0 - 1e-6)
        unlikelihood = -torch.log1p(-target_probs)
        reference_kl = (
            base_log_probs.exp() * (base_log_probs - adapted_log_probs)
        ).sum(dim=-1)
        per_token = unlikelihood + float(reference_beta) * reference_kl
        loss = (weights * per_token).sum()
        if backward:
            loss.backward()
        totals["loss"] += float(loss.detach().cpu())
        totals["unlikelihood"] += float((weights * unlikelihood).sum().detach().cpu())
        totals["reference_kl"] += float((weights * reference_kl).sum().detach().cpu())
        totals["target_logprob"] += float((weights * target_log_probs).sum().detach().cpu())
    return totals


def _calibrate_effect_scale(
    adapter: LowRankActionLogitAdapter,
    records: Sequence[ActionRecord],
    record_weights: Sequence[float],
    *,
    target_logprob_drop: float,
    reference_beta: float,
    microbatch_tokens: int,
    max_scale: float = 128.0,
    binary_steps: int = 18,
) -> Dict[str, Any]:
    """Find the smallest scale on the learned direction meeting the effect target."""
    baseline = _adapter_objective(
        adapter,
        records,
        record_weights,
        reference_beta=reference_beta,
        backward=False,
        microbatch_tokens=microbatch_tokens,
        adapter_scale=0.0,
    )

    def evaluate(scale: float) -> tuple[Dict[str, float], float]:
        stats = _adapter_objective(
            adapter,
            records,
            record_weights,
            reference_beta=reference_beta,
            backward=False,
            microbatch_tokens=microbatch_tokens,
            adapter_scale=scale,
        )
        delta = stats["target_logprob"] - baseline["target_logprob"]
        return stats, delta

    target = float(target_logprob_drop)
    if target <= 0.0:
        stats, delta = evaluate(1.0)
        return {
            "enabled": False,
            "target_logprob_drop": target,
            "selected_scale": 1.0,
            "achieved": True,
            "achieved_logprob_drop": -delta,
            "target_logprob_change": delta,
            "objective_at_selected_scale": stats,
            "bracket_evaluations": [{"scale": 1.0, "target_logprob_change": delta}],
            "binary_steps": 0,
        }

    lower = 0.0
    upper = 1.0
    upper_stats, upper_delta = evaluate(upper)
    bracket = [{"scale": upper, "target_logprob_change": upper_delta}]
    while upper_delta > -target and upper < max_scale:
        lower = upper
        upper = min(float(max_scale), 2.0 * upper)
        upper_stats, upper_delta = evaluate(upper)
        bracket.append({"scale": upper, "target_logprob_change": upper_delta})

    achieved = upper_delta <= -target
    used_binary_steps = 0
    if achieved:
        for _ in range(max(0, int(binary_steps))):
            middle = 0.5 * (lower + upper)
            middle_stats, middle_delta = evaluate(middle)
            used_binary_steps += 1
            if middle_delta <= -target:
                upper, upper_stats, upper_delta = middle, middle_stats, middle_delta
            else:
                lower = middle

    return {
        "enabled": True,
        "target_logprob_drop": target,
        "selected_scale": upper,
        "achieved": achieved,
        "achieved_logprob_drop": -upper_delta,
        "target_logprob_change": upper_delta,
        "objective_at_selected_scale": upper_stats,
        "bracket_evaluations": bracket,
        "binary_steps": used_binary_steps,
        "max_scale": float(max_scale),
    }


def _parameter_norm(module: nn.Module) -> float:
    values = [parameter.detach().float().square().sum() for parameter in module.parameters()]
    if not values:
        return 0.0
    return float(torch.sqrt(torch.stack(values).sum()).cpu())


def install_action_memory_patch(
    *,
    mode: str,
    rank: int = 4,
    lr: float = 1e-3,
    update_frequency: int = 5,
    window_size: int = 25,
    optimization_steps: int = 10,
    reference_beta: float = 1.0,
    apply_scale: float = 1.0,
    target_logprob_drop: float = 0.0,
    semantic_threshold: float = 0.85,
    microbatch_tokens: int = 8,
    seed: int = 42,
) -> None:
    if mode not in VALID_ACTION_MEMORY_MODES:
        raise ValueError(f"Unsupported action-memory mode: {mode}")
    if rank <= 0 or lr <= 0 or update_frequency <= 0 or window_size <= 0:
        raise ValueError("rank, lr, frequency, and window must be positive")
    if (
        optimization_steps <= 0
        or reference_beta < 0
        or apply_scale < 0
        or target_logprob_drop < 0
    ):
        raise ValueError("invalid optimizer/reference/apply settings")
    if not -1.0 <= semantic_threshold <= 1.0:
        raise ValueError("semantic threshold must be in [-1, 1]")

    original_init = FeedbackToPolicyTTTAgent.__init__
    original_act = FeedbackToPolicyTTTAgent._act
    original_observe_transition = FeedbackToPolicyTTTAgent.observe_transition
    original_save_memory = FeedbackToPolicyTTTAgent.save_memory
    original_load_memory = FeedbackToPolicyTTTAgent.load_memory

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.action_memory_mode = mode
        self.action_memory_rank = int(rank)
        self.action_memory_lr = float(lr)
        self.action_memory_update_frequency = int(update_frequency)
        self.action_memory_window_size = int(window_size)
        self.action_memory_optimization_steps = int(optimization_steps)
        self.action_memory_reference_beta = float(reference_beta)
        self.action_memory_apply_scale = float(apply_scale)
        self.action_memory_target_logprob_drop = float(target_logprob_drop)
        self.action_memory_semantic_threshold = float(semantic_threshold)
        self.action_memory_microbatch_tokens = int(microbatch_tokens)
        self.action_memory_seed = int(seed)
        self.action_memory_actions: List[str] = []
        self.action_memory_records: List[ActionRecord] = []
        self.action_memory_steps_seen = 0
        self.action_memory_last_update_step = -1
        self.action_memory_updates_total = 0
        self.action_memory_adapter = None
        self.action_memory_bootstrap_log = None
        self.action_memory_pending_record = None
        self.action_memory_last_trace: Dict[str, Any] = {}
        print(
            "[ActionMemory] "
            f"mode={mode} rank={rank} lr={lr} window={window_size} "
            f"frequency={update_frequency} opt_steps={optimization_steps} "
            f"reference_beta={reference_beta} apply_scale={apply_scale} "
            f"target_logprob_drop={target_logprob_drop} "
            f"semantic_threshold={semantic_threshold}",
            flush=True,
        )

    def ensure_action_adapter(self):
        if self.action_memory_mode not in PARAMETRIC_MODES:
            return None
        if self.action_memory_adapter is None:
            model = self.tlm.model
            config = model.config
            adapter = LowRankActionLogitAdapter(
                hidden_size=int(config.hidden_size),
                vocab_size=int(config.vocab_size),
                rank=self.action_memory_rank,
                seed=self.action_memory_seed + 90_001,
                protected_token_ids=_protected_json_token_ids(self.tlm.tokenizer),
            )
            adapter.to(device=self.tlm.device, dtype=torch.float32)
            self.action_memory_adapter = adapter
        return self.action_memory_adapter

    def reconstruct_bootstrap_records(self) -> Dict[str, Any] | None:
        if self.action_memory_mode not in PARAMETRIC_MODES:
            return None
        if self.action_memory_records:
            return self.action_memory_bootstrap_log
        source_log = getattr(self, "action_memory_source_agent_log", None)
        branch_step = int(getattr(self, "action_memory_branch_step", 0))
        if not source_log or not os.path.isfile(source_log):
            raise FileNotFoundError(f"Missing source agent log for action bootstrap: {source_log}")
        started = time.perf_counter()
        rows = []
        with open(source_log, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if int(row.get("step", -1)) < branch_step:
                    rows.append(row)
        rows = rows[-self.action_memory_window_size :]
        model = self.tlm.model
        model.eval()
        lm_head = model.get_output_embeddings()
        tokenizer = self.tlm.tokenizer
        records: List[ActionRecord] = []
        skipped = []
        max_length = int(getattr(self, "f2p_max_score_len", 4096))
        for row in rows:
            response = str(row.get("response") or "")
            span = find_json_action_span(response)
            trace = row.get("f2p_trace", {}) or {}
            prompt = trace.get("action_prompt")
            if span is None or not prompt:
                skipped.append({"step": row.get("step"), "reason": "missing_action_span_or_prompt"})
                continue
            messages = [
                {"role": "system", "content": self.f2p_system},
                {"role": "user", "content": prompt},
            ]
            chat = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            full_text = chat + response
            global_start = len(chat) + span[0]
            global_end = len(chat) + span[1]
            encoded = tokenizer(
                full_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            indices = [
                index
                for index, (left, right) in enumerate(encoded["offset_mapping"])
                if left >= global_start and right <= global_end and right > left
            ]
            if not indices or indices[0] <= 0:
                skipped.append({"step": row.get("step"), "reason": "no_full_action_tokens"})
                continue
            # Cut immediately after the action so logits_to_keep only returns
            # the small action suffix. Left truncation matches existing F2P.
            full_ids = encoded["input_ids"][: indices[-1] + 1]
            left_trim = max(0, len(full_ids) - max_length)
            action_indices = [index - left_trim for index in indices if index >= left_trim]
            input_ids = torch.tensor(
                full_ids[left_trim:], dtype=torch.long, device=self.tlm.device
            ).unsqueeze(0)
            token_count = len(action_indices)
            if not token_count or action_indices != list(
                range(input_ids.shape[1] - token_count, input_ids.shape[1])
            ):
                skipped.append({"step": row.get("step"), "reason": "non_suffix_action_tokens"})
                continue
            captured_hidden = None

            def capture_hidden(module, args):
                nonlocal captured_hidden
                captured_hidden = args[0][0, -(token_count + 1) : -1, :].detach()

            hook = lm_head.register_forward_pre_hook(capture_hidden)
            try:
                with torch.no_grad():
                    output = model(
                        input_ids=input_ids,
                        attention_mask=torch.ones_like(input_ids),
                        use_cache=False,
                        logits_to_keep=token_count + 1,
                    )
            finally:
                hook.remove()
            if captured_hidden is None or captured_hidden.shape[0] != token_count:
                skipped.append({"step": row.get("step"), "reason": "hidden_capture_failed"})
                continue
            records.append(
                ActionRecord(
                    action=str(row.get("action", "")),
                    hidden=captured_hidden.to("cpu", dtype=torch.float16),
                    base_logits=output.logits[0, :token_count, :].detach().to(
                        "cpu", dtype=torch.float16
                    ),
                    labels=input_ids[0, -token_count:].detach().to("cpu", dtype=torch.long),
                    source_step=int(row.get("step", -1)),
                    source="reconstructed_real_action_slot",
                )
            )
            del output, input_ids, captured_hidden
        self.action_memory_records = records[-self.action_memory_window_size :]
        self.action_memory_bootstrap_log = {
            "requested_records": len(rows),
            "reconstructed_records": len(self.action_memory_records),
            "reconstructed_action_tokens": sum(
                int(record.labels.numel()) for record in self.action_memory_records
            ),
            "skipped": skipped,
            "max_context_tokens": max_length,
            "seconds": time.perf_counter() - started,
        }
        return self.action_memory_bootstrap_log

    def maybe_fit_action_adapter(self) -> Dict[str, Any]:
        trace: Dict[str, Any] = {
            "updated": False,
            "step_before_action": self.action_memory_steps_seen,
            "history_size": len(self.action_memory_actions),
            "record_count": len(self.action_memory_records),
        }
        if self.action_memory_mode not in PARAMETRIC_MODES:
            trace["skip_reason"] = "non_parametric_mode"
            return trace
        bootstrap = reconstruct_bootstrap_records(self)
        if bootstrap is not None and self.action_memory_updates_total == 0:
            trace["bootstrap"] = bootstrap
        if not self.action_memory_records:
            trace["skip_reason"] = "no_action_slot_records"
            return trace
        if self.action_memory_steps_seen % self.action_memory_update_frequency != 0:
            trace["skip_reason"] = "not_update_boundary"
            return trace
        if self.action_memory_last_update_step == self.action_memory_steps_seen:
            trace["skip_reason"] = "already_updated"
            return trace

        started = time.perf_counter()
        adapter = ensure_action_adapter(self)
        adapter.reset_parameters_to_initial()
        adapter.train()
        records, weights, clustering = _weighted_record_plan(self)
        optimizer = AdamW(adapter.parameters(), lr=self.action_memory_lr, weight_decay=0.0)
        before = _adapter_objective(
            adapter,
            records,
            weights,
            reference_beta=self.action_memory_reference_beta,
            backward=False,
            microbatch_tokens=self.action_memory_microbatch_tokens,
        )
        grad_norms = []
        for _ in range(self.action_memory_optimization_steps):
            optimizer.zero_grad(set_to_none=True)
            _adapter_objective(
                adapter,
                records,
                weights,
                reference_beta=self.action_memory_reference_beta,
                backward=True,
                microbatch_tokens=self.action_memory_microbatch_tokens,
            )
            grad_norm_sq = sum(
                float(parameter.grad.detach().float().square().sum().cpu())
                for parameter in adapter.parameters()
                if parameter.grad is not None
            )
            grad_norms.append(math.sqrt(grad_norm_sq))
            optimizer.step()
        adapter.eval()
        after = _adapter_objective(
            adapter,
            records,
            weights,
            reference_beta=self.action_memory_reference_beta,
            backward=False,
            microbatch_tokens=self.action_memory_microbatch_tokens,
        )
        calibration = _calibrate_effect_scale(
            adapter,
            records,
            weights,
            target_logprob_drop=self.action_memory_target_logprob_drop,
            reference_beta=self.action_memory_reference_beta,
            microbatch_tokens=self.action_memory_microbatch_tokens,
        )
        if self.action_memory_target_logprob_drop > 0.0:
            self.action_memory_apply_scale = float(calibration["selected_scale"])
        self.action_memory_last_update_step = self.action_memory_steps_seen
        self.action_memory_updates_total += 1
        trace.update(
            {
                "updated": True,
                "selected_record_count": len(records),
                "selected_action_token_count": sum(int(record.labels.numel()) for record in records),
                "selected_actions": [record.action for record in records],
                "record_weights": weights,
                **clustering,
                "objective_before": before,
                "objective_after": after,
                "target_logprob_change": after["target_logprob"] - before["target_logprob"],
                "effect_calibration": calibration,
                "generation_apply_scale": self.action_memory_apply_scale,
                "adapter_parameter_norm": _parameter_norm(adapter),
                "grad_norms": grad_norms,
                "updates_total": self.action_memory_updates_total,
                "seconds": time.perf_counter() - started,
                "loss_definition": (
                    "negative_unlikelihood(action tokens at real JSON action slots) "
                    f"+ {self.action_memory_reference_beta} * KL(base || adapted)"
                ),
                "reset_before_each_update": True,
            }
        )
        return trace

    def generate_with_action_adapter(self, *, user_prompt: str):
        adapter = ensure_action_adapter(self)
        adapter.eval()
        model = self.tlm.model
        processor = _ActionSlotProcessor(
            agent=self, adapter=adapter, scale=self.action_memory_apply_scale
        )
        lm_head = model.get_output_embeddings()
        hook = lm_head.register_forward_pre_hook(processor.capture_lm_head_input)
        original_model_generate = model.generate
        generated_ids = None

        def injected_generate(*args, **kwargs):
            nonlocal generated_ids
            processors = kwargs.get("logits_processor")
            if processors is None:
                processors = LogitsProcessorList()
            elif not isinstance(processors, LogitsProcessorList):
                processors = LogitsProcessorList(list(processors))
            processors.append(processor)
            kwargs["logits_processor"] = processors
            generated_ids = original_model_generate(*args, **kwargs)
            return generated_ids

        model.generate = injected_generate
        try:
            result = self.tlm.generate(user_prompt=user_prompt, system_prompt=self.f2p_system)
        finally:
            model.generate = original_model_generate
            hook.remove()
        return result, processor, generated_ids

    def _build_action_prompt(self, obs: Dict[str, Any], *, include_action_history: bool) -> str:
        obs_text = obs.get("text", "")
        memory_text = "\n\n".join(self.short_term_memory)
        prompt = (
            "My Current Observation:\n"
            + obs_text
            + ("\n\nVerified recent feedback:\n" + memory_text if memory_text else "")
        )
        if include_action_history:
            history_prompt = build_action_history_prompt(self.action_memory_actions)
            if history_prompt:
                prompt += "\n\n" + history_prompt
        return prompt + "\n\nChoose one valid action and briefly predict its immediate environment change."

    def _parse_generated_action(self, lm_output):
        response = lm_output.get("response") if isinstance(lm_output, dict) else None
        parsed = self.cfg.json_parser(response or "")
        if not isinstance(parsed, dict):
            parsed = {"action": "wait", "predicted_outcome": "", "reasoning": "parse failure"}
        action = parsed.get("action", "wait")
        if not isinstance(action, str) or not action.strip():
            action = "wait"
        self._last_action = action.strip()
        self._last_prediction = str(parsed.get("predicted_outcome", ""))
        prior = self._score_action("", self._last_action, requires_grad=False)
        self._last_action_prior = float(prior.detach().cpu()) if torch.is_tensor(prior) else float(prior)
        return self._last_action

    def patched_act(self, obs: Dict[str, Any]):
        if self.action_memory_mode == "control":
            self.action_memory_last_trace = {
                "mode": self.action_memory_mode,
                "step_before_action": self.action_memory_steps_seen,
                "history_size": len(self.action_memory_actions),
                "updated": False,
                "prompt_history_count": 0,
            }
            return original_act(self, obs)

        include_prompt = self.action_memory_mode == "prompt_history"
        self._last_action_prompt = self._build_action_memory_prompt(
            obs, include_action_history=include_prompt
        )
        update_trace = maybe_fit_action_adapter(self)
        if self.action_memory_mode in PARAMETRIC_MODES:
            lm_output, processor, generated_ids = generate_with_action_adapter(
                self, user_prompt=self._last_action_prompt
            )
            action = _parse_generated_action(self, lm_output)
            if generated_ids is not None:
                processor.finalize(
                    generated_ids,
                    action=action,
                    source_step=self.action_memory_steps_seen,
                )
            self.action_memory_pending_record = processor.record
            generation_trace = {
                "action_slot_gate_calls": processor.applied_positions,
                "captured_action_tokens": (
                    int(processor.record.labels.numel()) if processor.record is not None else 0
                ),
                "adapter_apply_scale": self.action_memory_apply_scale,
                **processor.effect_trace,
            }
        else:
            lm_output = self.tlm.generate(
                user_prompt=self._last_action_prompt, system_prompt=self.f2p_system
            )
            action = _parse_generated_action(self, lm_output)
            generation_trace = {
                "action_slot_gate_calls": 0,
                "captured_action_tokens": 0,
                "adapter_apply_scale": 0.0,
            }
        self.action_memory_last_trace = {
            "mode": self.action_memory_mode,
            "step_before_action": self.action_memory_steps_seen,
            "history_size": len(self.action_memory_actions),
            "history_before_action": list(self.action_memory_actions),
            "prompt_history_count": len(self.action_memory_actions) if include_prompt else 0,
            "update": update_trace,
            "generation": generation_trace,
        }
        return (
            action,
            lm_output["num_input_tokens"],
            lm_output["num_output_tokens"],
            lm_output["response"],
        )

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
        prior_actions = list(self.action_memory_actions)
        exact_repeat = str(action) in prior_actions
        semantic_repeat = None
        semantic_similarity = None
        if prior_actions:
            embedding_layer = self.tlm.model.get_input_embeddings()
            chosen_embedding = frozen_action_embedding(
                self.tlm.tokenizer, embedding_layer, str(action)
            )
            history_embeddings = [
                frozen_action_embedding(self.tlm.tokenizer, embedding_layer, item)
                for item in prior_actions
            ]
            semantic_similarity = max(
                float(torch.dot(chosen_embedding, candidate))
                for candidate in history_embeddings
            )
            semantic_repeat = semantic_similarity >= self.action_memory_semantic_threshold

        self.action_memory_actions.append(str(action))
        self.action_memory_actions = self.action_memory_actions[-self.action_memory_window_size :]
        if self.action_memory_pending_record is not None:
            self.action_memory_records.append(self.action_memory_pending_record)
            self.action_memory_records = self.action_memory_records[-self.action_memory_window_size :]
            self.action_memory_pending_record = None
        self.action_memory_steps_seen += 1
        self.action_memory_last_trace.update(
            {
                "chosen_action": str(action),
                "chosen_exact_repeat": exact_repeat,
                "chosen_semantic_repeat": semantic_repeat,
                "chosen_max_semantic_similarity": semantic_similarity,
                "history_after_action": list(self.action_memory_actions),
                "steps_seen_after_action": self.action_memory_steps_seen,
                "record_count_after_action": len(self.action_memory_records),
            }
        )
        self.f2p_last_trace = dict(self.f2p_last_trace)
        self.f2p_last_trace["action_memory"] = dict(self.action_memory_last_trace)
        if self.f2p_log_path:
            path = os.path.join(
                os.path.dirname(self.f2p_log_path), "action_memory_intermediates.jsonl"
            )
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(self.action_memory_last_trace, ensure_ascii=False) + "\n")

    def patched_save_memory(self, full_memory_dir: str) -> None:
        original_save_memory(self, full_memory_dir)
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data["action_memory_branch"] = {
            "mode": self.action_memory_mode,
            "rank": self.action_memory_rank,
            "lr": self.action_memory_lr,
            "update_frequency": self.action_memory_update_frequency,
            "window_size": self.action_memory_window_size,
            "optimization_steps": self.action_memory_optimization_steps,
            "reference_beta": self.action_memory_reference_beta,
            "apply_scale": self.action_memory_apply_scale,
            "target_logprob_drop": self.action_memory_target_logprob_drop,
            "semantic_threshold": self.action_memory_semantic_threshold,
            "microbatch_tokens": self.action_memory_microbatch_tokens,
            "seed": self.action_memory_seed,
            "action_history": list(self.action_memory_actions),
            "steps_seen": self.action_memory_steps_seen,
            "last_update_step": self.action_memory_last_update_step,
            "updates_total": self.action_memory_updates_total,
            "source_agent_log": getattr(self, "action_memory_source_agent_log", None),
            "branch_step": getattr(self, "action_memory_branch_step", None),
            "last_trace": self.action_memory_last_trace,
        }
        atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))
        if self.action_memory_adapter is not None:
            torch.save(
                self.action_memory_adapter.state_dict(),
                os.path.join(full_memory_dir, "action_logit_adapter.pt"),
            )
        if self.action_memory_records:
            torch.save(
                self.action_memory_records,
                os.path.join(full_memory_dir, "action_slot_records.pt"),
            )

    def patched_load_memory(self, full_memory_dir: str) -> None:
        original_load_memory(self, full_memory_dir)
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        if not os.path.isfile(path):
            return
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        state = data.get("action_memory_branch", {}) or {}
        saved_mode = state.get("mode")
        # The prepared step100 checkpoint deliberately has no mode yet.
        if saved_mode and saved_mode != "uninitialized" and saved_mode != self.action_memory_mode:
            raise ValueError(
                f"Saved action-memory mode {saved_mode!r} != requested {self.action_memory_mode!r}"
            )
        self.action_memory_actions = list(state.get("action_history", []) or [
        ])[-self.action_memory_window_size :]
        self.action_memory_steps_seen = int(state.get("steps_seen", 0))
        self.action_memory_last_update_step = int(state.get("last_update_step", -1))
        self.action_memory_updates_total = int(state.get("updates_total", 0))
        saved_target = state.get("target_logprob_drop")
        if saved_target is not None and not math.isclose(
            float(saved_target), self.action_memory_target_logprob_drop
        ):
            raise ValueError(
                f"Saved target log-prob drop {saved_target} != requested "
                f"{self.action_memory_target_logprob_drop}"
            )
        if "apply_scale" in state and saved_mode != "uninitialized":
            self.action_memory_apply_scale = float(state["apply_scale"])
        self.action_memory_source_agent_log = state.get("source_agent_log")
        self.action_memory_branch_step = int(state.get("branch_step", 0) or 0)
        self.action_memory_last_trace = dict(state.get("last_trace", {}) or {})
        records_path = os.path.join(full_memory_dir, "action_slot_records.pt")
        if os.path.isfile(records_path):
            self.action_memory_records = torch.load(records_path, map_location="cpu", weights_only=False)
        adapter_path = os.path.join(full_memory_dir, "action_logit_adapter.pt")
        if self.action_memory_mode in PARAMETRIC_MODES and os.path.isfile(adapter_path):
            adapter = ensure_action_adapter(self)
            adapter.load_state_dict(torch.load(adapter_path, map_location=self.tlm.device, weights_only=True))
            adapter.eval()

    FeedbackToPolicyTTTAgent.__init__ = patched_init
    FeedbackToPolicyTTTAgent._ensure_action_adapter = ensure_action_adapter
    FeedbackToPolicyTTTAgent._reconstruct_action_memory_records = reconstruct_bootstrap_records
    FeedbackToPolicyTTTAgent._maybe_fit_action_adapter = maybe_fit_action_adapter
    FeedbackToPolicyTTTAgent._generate_with_action_adapter = generate_with_action_adapter
    FeedbackToPolicyTTTAgent._build_action_memory_prompt = _build_action_prompt
    FeedbackToPolicyTTTAgent._act = patched_act
    FeedbackToPolicyTTTAgent.observe_transition = patched_observe_transition
    FeedbackToPolicyTTTAgent.save_memory = patched_save_memory
    FeedbackToPolicyTTTAgent.load_memory = patched_load_memory
