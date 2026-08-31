from __future__ import annotations

import json
import math
import os
from functools import lru_cache
from typing import Any, Dict, List, Type

import torch

from agents.parametric.environment_prediction_ttt_agent import (
    EnvironmentPredictionTTTAgent,
)
from utils import atomic_write


class HindsightLongHorizonTTTAgent(EnvironmentPredictionTTTAgent):
    """Environment-prediction TTT plus real-future hindsight policy TTT.

    The environment loss is inherited unchanged.  After a real future window
    is available, the same LoRA scores the executed action with and without the
    observed future.  The difference is multiplied by the signed environment
    feedback and used as the coefficient of the action log-likelihood loss.

    This implementation deliberately has no value model, candidate search,
    second adapter, or imagined transition in the training data.
    """

    def __init__(self, id: str, name: str, cfg=None, train_cfg=None):
        super().__init__(id=id, name=name, cfg=cfg, train_cfg=train_cfg)
        self.hindsight_log_path: str | None = None
        self.hindsight_horizon = int(getattr(self.cfg, "hindsight_horizon", 4))
        self.policy_update_frequency = int(getattr(self.cfg, "policy_update_frequency", 5))
        self.policy_lr = float(getattr(self.cfg, "policy_lr", self.lr))
        self.policy_epochs = int(getattr(self.cfg, "policy_epochs", 1))
        self.policy_batch_size = 1
        self._trajectory_buffer: List[Dict[str, Any]] = []
        self._credit_buffer: List[Dict[str, Any]] = []
        self._last_hindsight_trace: Dict[str, Any] = {}
        self._environment_steps = 0
        self.hindsight_credits_total = 0
        self.policy_updates_total = 0
        self.policy_update_steps_total = 0
        self.incomplete_windows_total = 0
        # Keep verified text memory useful without allowing five full game
        # observations to exhaust the model context during long episodes.
        self.prompt_memory_max_chars = int(
            getattr(self.cfg, "prompt_memory_max_chars", 12000)
        )
        self._current_action_prompt = ""
        self._current_action = ""

    def _bounded_prompt_memory(self) -> List[str]:
        """Return recent verified memories under a deterministic text budget."""
        budget = max(0, int(self.prompt_memory_max_chars))
        if budget == 0:
            return []
        kept: List[str] = []
        used = 0
        # Retain the newest items first; preserve chronological order in the
        # final prompt after selecting them.
        for item in reversed(self.short_term_memory):
            remaining = budget - used
            if remaining <= 0:
                break
            clipped = item[-remaining:]
            kept.append(clipped)
            used += len(clipped)
        return list(reversed(kept))

    @staticmethod
    def _reward_dict(reward: Any) -> Dict[str, float]:
        if reward is None:
            return {}
        if isinstance(reward, dict):
            return {str(k): float(v or 0) for k, v in reward.items()}
        data = getattr(reward, "__dict__", {}) or {}
        return {str(k): float(v or 0) for k, v in data.items()}

    @classmethod
    def _signed_feedback(cls, reward: Any) -> float:
        """A transparent signed proxy from environment feedback only."""
        r = cls._reward_dict(reward)
        positive = sum(r.get(k, 0.0) for k in (
            "quest", "exploration", "craft", "unique_kill", "kill",
            "side_quest", "trade",
        ))
        return float(positive - r.get("death", 0.0))

    def _act(self, obs: Dict[str, Any]):
        original_memory = self.short_term_memory
        self.short_term_memory = self._bounded_prompt_memory()
        try:
            action, n_in, n_out, response = super()._act(obs)
        finally:
            self.short_term_memory = original_memory
        self._current_action_prompt = self._last_prompt
        self._current_action = action
        return action, n_in, n_out, response

    def _score_action(self, prompt: str, action: str) -> float:
        """Length-normalized conditional log-probability of an action string."""
        peft_m = self.tlm._ensure_peft()
        tokenizer = self.tlm.tokenizer
        prefix = prompt + "\n\nAction actually taken:\n"
        p_ids = tokenizer(prefix, add_special_tokens=True, truncation=False)["input_ids"]
        a_ids = tokenizer(action, add_special_tokens=False, truncation=False)["input_ids"]
        if not a_ids:
            return 0.0
        max_len = max(32, int(self.max_seq_len))
        if len(a_ids) >= max_len:
            a_ids = a_ids[: max_len - 1]
        keep_prompt = max(1, max_len - len(a_ids))
        p_ids = p_ids[-keep_prompt:]
        ids = p_ids + a_ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.tlm.device)
        attention = torch.ones_like(input_ids)
        peft_m.eval()
        with torch.inference_mode():
            with torch.amp.autocast(
                "cuda", enabled=(self.fp16 and torch.cuda.is_available())
            ):
                out = peft_m(input_ids=input_ids, attention_mask=attention)
        # Logits at position p-1 predict the first action token.
        start = max(0, len(p_ids) - 1)
        logits = out.logits[:, start : start + len(a_ids), :].float()
        labels = torch.tensor([a_ids], dtype=torch.long, device=self.tlm.device)
        logp = torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return float(logp.mean().detach().cpu())

    def _future_text(self, window: List[Dict[str, Any]]) -> str:
        parts = []
        for i, item in enumerate(window):
            parts.append(
                f"Future transition {i}:\n"
                f"Action taken:\n{item.get('action', '')}\n"
                f"Real next observation:\n{item.get('next_observation', '')}"
            )
        return "\n\n".join(parts)

    def _append_credit_log(self, record: Dict[str, Any]) -> None:
        if not self.hindsight_log_path:
            return
        os.makedirs(os.path.dirname(self.hindsight_log_path), exist_ok=True)
        with open(self.hindsight_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _train_action_policy(self) -> Dict[str, Any]:
        if not self._credit_buffer:
            return {"triggered": False, "steps": 0, "loss": None}
        peft_m = self.tlm._ensure_peft()
        peft_m.train()
        optimizer = torch.optim.AdamW(peft_m.parameters(), lr=self.policy_lr)
        scaler = torch.amp.GradScaler(
            "cuda", enabled=(self.fp16 and torch.cuda.is_available())
        )
        losses = []
        steps = 0
        # One small batch keeps memory bounded and makes every policy update auditable.
        for _ in range(max(1, self.policy_epochs)):
            for item in self._credit_buffer:
                prompt = item["prompt"] + "\n\nAction actually taken:\n"
                tokenizer = self.tlm.tokenizer
                p_ids = tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
                a_ids = tokenizer(item["action"], add_special_tokens=False, truncation=False)["input_ids"]
                if not a_ids:
                    continue
                max_len = max(32, int(self.max_seq_len))
                keep_prompt = max(1, max_len - min(len(a_ids), max_len - 1))
                p_ids = p_ids[-keep_prompt:]
                a_ids = a_ids[: max_len - len(p_ids)]
                ids = p_ids + a_ids
                x = torch.tensor([ids], dtype=torch.long, device=self.tlm.device)
                mask = torch.ones_like(x)
                with torch.amp.autocast(
                    "cuda", enabled=(self.fp16 and torch.cuda.is_available())
                ):
                    out = peft_m(input_ids=x, attention_mask=mask)
                    logits = out.logits[:, len(p_ids) - 1 : len(p_ids) - 1 + len(a_ids), :]
                    labels = torch.tensor([a_ids], dtype=torch.long, device=self.tlm.device)
                    token_nll = -torch.log_softmax(logits.float(), dim=-1).gather(
                        -1, labels.unsqueeze(-1)
                    ).squeeze(-1).mean()
                    weighted = item["advantage"] * token_nll / self.grad_accum
                scaler.scale(weighted).backward()
                losses.append(float(weighted.detach().cpu()))
                steps += 1
                if steps % max(1, self.grad_accum) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(peft_m.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
        peft_m.eval()
        self.policy_updates_total += 1
        self.policy_update_steps_total += steps
        self._credit_buffer = []
        return {
            "triggered": True,
            "steps": steps,
            "loss": (sum(losses) / len(losses)) if losses else None,
        }

    def _finalize_one(self, window: List[Dict[str, Any]], incomplete: bool = False) -> None:
        if len(window) < 2:
            return
        item = window[0]
        future_text = self._future_text(window)
        hindsight_prompt = (
            item["prompt"]
            + "\n\nReal future observed after the action:\n"
            + future_text
        )
        prior = self._score_action(item["prompt"], item["action"])
        hindsight = self._score_action(hindsight_prompt, item["action"])
        evidence = hindsight - prior
        feedback = sum(self._signed_feedback(x.get("reward")) for x in window)
        advantage = feedback * evidence
        record = {
            "source_step": item["step"],
            "effective_horizon": len(window) - 1,
            "configured_horizon": self.hindsight_horizon,
            "action": item["action"],
            "prior_logprob": prior,
            "hindsight_logprob": hindsight,
            "evidence_e_t": evidence,
            "feedback_R": feedback,
            "advantage_A_t": advantage,
            "reward_window": [self._reward_dict(x.get("reward")) for x in window],
            "future_text": future_text,
            "incomplete_window": bool(incomplete),
            "policy_updates_before": self.policy_updates_total,
        }
        self.hindsight_credits_total += 1
        if incomplete:
            self.incomplete_windows_total += 1
        self._credit_buffer.append({
            "prompt": item["prompt"],
            "action": item["action"],
            "advantage": float(advantage),
        })
        policy = {"triggered": False, "steps": 0, "loss": None}
        if len(self._credit_buffer) >= max(1, self.policy_update_frequency):
            policy = self._train_action_policy()
        record["policy_update_triggered"] = policy["triggered"]
        record["policy_update_steps"] = policy["steps"]
        record["policy_loss"] = policy["loss"]
        record["policy_updates_after"] = self.policy_updates_total
        self._append_credit_log(record)
        self._last_hindsight_trace = {
            "source_step": item["step"],
            "effective_horizon": len(window) - 1,
            "prior_logprob": prior,
            "hindsight_logprob": hindsight,
            "evidence_e_t": evidence,
            "feedback_R": feedback,
            "advantage_A_t": advantage,
            "incomplete_window": bool(incomplete),
            "policy_update_triggered": policy["triggered"],
            "policy_loss": policy["loss"],
            "hindsight_credits_total": self.hindsight_credits_total,
            "policy_updates_total": self.policy_updates_total,
            "policy_update_steps_total": self.policy_update_steps_total,
            "pending_windows": len(self._trajectory_buffer),
        }

    def observe_transition(self, previous_obs, action: str, next_obs, reward=None, info=None):
        before = self.steps_trained_total
        super().observe_transition(
            previous_obs, action, next_obs, reward=reward, info=info
        )
        prev_text = previous_obs.get("text", "") if isinstance(previous_obs, dict) else str(previous_obs)
        next_text = next_obs.get("text", "") if isinstance(next_obs, dict) else str(next_obs)
        self._trajectory_buffer.append({
            "step": self._environment_steps,
            "prompt": self._current_action_prompt,
            "action": action,
            "previous_observation": prev_text,
            "next_observation": next_text,
            "reward": self._reward_dict(reward),
        })
        self._environment_steps += 1
        if len(self._trajectory_buffer) >= self.hindsight_horizon + 1:
            window = self._trajectory_buffer[: self.hindsight_horizon + 1]
            self._finalize_one(window, incomplete=False)
            self._trajectory_buffer.pop(0)
        self._last_hindsight_trace["env_training_steps_before"] = before
        self._last_hindsight_trace["env_training_steps_after"] = self.steps_trained_total

    def finish_episode(self):
        # Use shorter real windows at the end rather than silently losing them.
        while len(self._trajectory_buffer) > 1:
            window = self._trajectory_buffer[: min(self.hindsight_horizon + 1, len(self._trajectory_buffer))]
            self._finalize_one(window, incomplete=(len(window) - 1 < self.hindsight_horizon))
            self._trajectory_buffer.pop(0)
        if self._credit_buffer:
            policy = self._train_action_policy()
            self._last_hindsight_trace["episode_end_policy_update"] = policy

    def get_hindsight_trace(self):
        trace = dict(self._last_hindsight_trace)
        trace["pending_windows"] = len(self._trajectory_buffer)
        trace["hindsight_credits_total"] = self.hindsight_credits_total
        trace["policy_updates_total"] = self.policy_updates_total
        trace["policy_update_steps_total"] = self.policy_update_steps_total
        trace["incomplete_windows_total"] = self.incomplete_windows_total
        return trace

    def save_memory(self, full_memory_dir: str) -> None:
        super().save_memory(full_memory_dir)
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data["hindsight"] = {
            "hindsight_horizon": self.hindsight_horizon,
            "policy_update_frequency": self.policy_update_frequency,
            "policy_lr": self.policy_lr,
            "policy_epochs": self.policy_epochs,
            "hindsight_credits_total": self.hindsight_credits_total,
            "policy_updates_total": self.policy_updates_total,
            "policy_update_steps_total": self.policy_update_steps_total,
            "incomplete_windows_total": self.incomplete_windows_total,
            "environment_steps": self._environment_steps,
            "pending_windows": self._trajectory_buffer,
        }
        atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))

    def load_memory(self, full_memory_dir: str) -> None:
        super().load_memory(full_memory_dir)
        path = os.path.join(full_memory_dir, self.memory_paths[0])
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        h = data.get("hindsight", {}) or {}
        self.hindsight_horizon = int(h.get("hindsight_horizon", self.hindsight_horizon))
        self.policy_update_frequency = int(h.get("policy_update_frequency", self.policy_update_frequency))
        self.policy_lr = float(h.get("policy_lr", self.policy_lr))
        self.policy_epochs = int(h.get("policy_epochs", self.policy_epochs))
        self.hindsight_credits_total = int(h.get("hindsight_credits_total", 0))
        self.policy_updates_total = int(h.get("policy_updates_total", 0))
        self.policy_update_steps_total = int(h.get("policy_update_steps_total", 0))
        self.incomplete_windows_total = int(h.get("incomplete_windows_total", 0))
        self._environment_steps = int(h.get("environment_steps", 0))
        self._trajectory_buffer = list(h.get("pending_windows", []) or [])


@lru_cache(maxsize=None)
def create_hindsight_long_horizon_ttt_agent(Agent: Type):
    class_name = f"HindsightLongHorizonTTTAgent__{Agent.__module__}.{Agent.__name__}"
    return type(
        class_name,
        (HindsightLongHorizonTTTAgent, Agent),
        {"__module__": Agent.__module__, "__agent__": Agent},
    )
