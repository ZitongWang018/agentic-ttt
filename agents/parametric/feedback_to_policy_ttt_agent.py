from __future__ import annotations

import hashlib
import json
import math
import os
import re
from functools import lru_cache
from typing import Any, Dict, List, Type

import torch
from torch.optim import AdamW

from agents.parametric.lora_sft_agent import LoRASFTAgent


class FeedbackToPolicyTTTAgent(LoRASFTAgent):
    """Single-LoRA hindsight feedback-to-policy adaptation.

    The action path remains the original sampled policy. After the real
    transition arrives, the agent scores the exact executed action under the
    predicted and real outcome contexts and applies the signed F2P loss:
        L = -tanh(beta * (ell_real - ell_pred)) * log pi(a | H)
    No reward, Q-value, candidate search, KL distillation, or outcome loss is
    added. All intermediate values are persisted for post-hoc analysis.
    """

    def __init__(self, id: str, name: str, cfg=None, train_cfg=None):
        super().__init__(id=id, name=name, cfg=cfg, train_cfg=train_cfg)
        self.short_term_memory: List[str] = []
        self.f2p_buffer: List[Dict[str, Any]] = []
        self.f2p_last_trace: Dict[str, Any] = {}
        self.f2p_log_path: str | None = None
        self.f2p_beta = float(getattr(self.cfg, "f2p_beta", 1.0))
        self.f2p_update_frequency = int(getattr(self.cfg, "f2p_update_frequency", 5))
        self.f2p_max_score_len = int(getattr(self.cfg, "max_seq_len", 4096))
        self.f2p_system = (
            self.cfg.system_prompt
            + "\nIn addition to the required reasoning and action keys, include a short predicted_outcome key describing the immediate environment change."
        )
        self._last_action_prompt = ""
        self._last_prediction = ""
        self._last_action = ""
        self._last_action_prior = None

    def _chat_prompt(self, outcome: str = "", base_prompt: str | None = None) -> str:
        user = base_prompt if base_prompt is not None else self._last_action_prompt
        if outcome:
            user += "\n\nObserved environment outcome (do not infer beyond this):\n" + outcome
        user += "\n\nExact executed action to score:\n"
        messages = [
            {"role": "system", "content": self.f2p_system},
            {"role": "user", "content": user},
        ]
        return self.tlm.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def _score_action(self, outcome: str, action: str, requires_grad: bool = False, base_prompt: str | None = None) -> float | torch.Tensor:
        """Teacher-force only the exact action string; do not score reasoning/JSON."""
        prompt_text = self._chat_prompt(outcome, base_prompt=base_prompt)
        tok = self.tlm.tokenizer
        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(prompt_text + action, add_special_tokens=False)["input_ids"]
        action_start = len(prompt_ids)
        if len(full_ids) <= action_start:
            return torch.tensor(0.0, device=self.tlm.device, requires_grad=requires_grad)
        keep = min(len(full_ids), self.f2p_max_score_len)
        start = max(0, len(full_ids) - keep)
        # Preserve the action suffix; truncate only the left context.
        input_ids = torch.tensor(full_ids[start:], dtype=torch.long, device=self.tlm.device).unsqueeze(0)
        model = self.tlm.model
        context_shift = max(0, action_start - start - 1)
        if not requires_grad:
            model.eval()
        with torch.set_grad_enabled(requires_grad):
            out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))
            logits = out.logits[:, :-1, :].float()
            labels = input_ids[:, 1:]
            first = min(context_shift, logits.shape[1])
            action_logits = logits[:, first:, :]
            action_labels = labels[:, first:]
            if action_labels.numel() == 0:
                return torch.tensor(0.0, device=self.tlm.device, requires_grad=requires_grad)
            logp = torch.log_softmax(action_logits, dim=-1).gather(-1, action_labels.unsqueeze(-1)).squeeze(-1)
            # Length-normalized action log-probability.
            return logp.mean()

    @staticmethod
    def _state_delta(previous: str, current: str, action: str = "") -> Dict[str, Any]:
        def grab(text, pattern, default=""):
            m = re.search(pattern, text or "")
            return m.group(1).strip() if m else default

        def intgrab(text, pattern):
            m = re.search(pattern, text or "")
            return int(m.group(1)) if m else None

        def line_set(text):
            return [x.strip() for x in (text or "").splitlines() if x.strip()]

        loc0 = grab(previous, r"Current Location:\s*([^\n]+)")
        loc1 = grab(current, r"Current Location:\s*([^\n]+)")
        h0 = intgrab(previous, r"My health is at\s*(\d+)")
        h1 = intgrab(current, r"My health is at\s*(\d+)")
        xp0 = intgrab(previous, r"My experience is at\s*(\d+)")
        xp1 = intgrab(current, r"My experience is at\s*(\d+)")
        events = []
        action_echo = re.compile(r"^(i|you)\s+(picked up|pick up|entered|enter|talked to|talk to|inspected|inspect|crafted|craft|equipped|equip|attacked|attack|defended|defend|waited|wait|wrote|write|stored|store|dropped|drop|invoked|invoke)\b", re.I)
        for line in line_set(current):
            # Remove direct textual echoes of the just-executed action from the
            # posterior condition; keep state changes and environment errors.
            if action_echo.search(line):
                continue
            if re.search(r"^(I |You |Stage completed:|Cannot |I was |The |An )", line, re.I):
                events.append(line[:500])
        return {
            "location_before": loc0, "location_after": loc1,
            "location_changed": bool(loc0 and loc1 and loc0 != loc1),
            "health_before": h0, "health_after": h1,
            "health_delta": (h1 - h0) if h0 is not None and h1 is not None else None,
            "experience_before": xp0, "experience_after": xp1,
            "experience_delta": (xp1 - xp0) if xp0 is not None and xp1 is not None else None,
            "stage_completed": [x for x in events if x.lower().startswith("stage completed")],
            "event_lines": events[:12],
        }

    @staticmethod
    def _outcome_text(delta: Dict[str, Any]) -> str:
        return json.dumps(delta, ensure_ascii=False, sort_keys=True)

    def _append_trace(self, trace: Dict[str, Any]):
        if not self.f2p_log_path:
            return
        os.makedirs(os.path.dirname(self.f2p_log_path), exist_ok=True)
        with open(self.f2p_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    def _act(self, obs: Dict[str, Any]):
        obs_text = obs.get("text", "")
        memory_text = "\n\n".join(self.short_term_memory)
        self._last_action_prompt = (
            "My Current Observation:\n" + obs_text
            + ("\n\nVerified recent feedback:\n" + memory_text if memory_text else "")
            + "\n\nChoose one valid action and briefly predict its immediate environment change."
        )
        lm_output = self.tlm.generate(user_prompt=self._last_action_prompt, system_prompt=self.f2p_system)
        parsed = self.cfg.json_parser(lm_output["response"])
        if not isinstance(parsed, dict):
            parsed = {"action": "wait", "predicted_outcome": "", "reasoning": "parse failure"}
        action = parsed.get("action", "wait")
        if not isinstance(action, str) or not action.strip():
            action = "wait"
        self._last_action = action.strip()
        self._last_prediction = str(parsed.get("predicted_outcome", ""))
        # Prior score is logged; no candidate is generated or reranked.
        prior = self._score_action("", self._last_action, requires_grad=False)
        self._last_action_prior = float(prior.detach().cpu()) if torch.is_tensor(prior) else float(prior)
        return self._last_action, lm_output["num_input_tokens"], lm_output["num_output_tokens"], lm_output["response"]

    def _policy_update(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not batch:
            return {"updated": False}
        peft = self.tlm._ensure_peft()
        peft.train()
        opt = AdamW(peft.parameters(), lr=self.lr)
        opt.zero_grad(set_to_none=True)
        losses, current_scores = [], []
        for item in batch:
            score = self._score_action("", item["action"], requires_grad=True, base_prompt=item["prompt"])
            current_scores.append(float(score.detach().cpu()))
            # This is exactly the requested signed F2P loss; w is detached.
            loss = -float(item["w_t"]) * score
            losses.append(float(loss.detach().cpu()))
            # Sequential backward gives the same gradient as the batch mean,
            # without retaining one full computation graph per item.
            (loss / len(batch)).backward()
        total_value = sum(losses) / len(losses)
        torch.nn.utils.clip_grad_norm_(peft.parameters(), max_norm=1.0)
        opt.step()
        peft.eval()
        self.steps_trained_total += 1
        return {"updated": True, "loss": total_value, "current_action_scores": current_scores, "batch_size": len(batch)}

    def observe_transition(self, previous_obs, action: str, next_obs, reward=None, info=None):
        prev = previous_obs.get("text", "") if isinstance(previous_obs, dict) else str(previous_obs)
        nxt = next_obs.get("text", "") if isinstance(next_obs, dict) else str(next_obs)
        delta = self._state_delta(prev, nxt, action=action)
        real_outcome = self._outcome_text(delta)
        pred_score = self._score_action(self._last_prediction, action, requires_grad=False)
        real_score = self._score_action(real_outcome, action, requires_grad=False)
        prior = float(self._last_action_prior if self._last_action_prior is not None else self._score_action("", action, False))
        pred_score = float(pred_score.detach().cpu()) if torch.is_tensor(pred_score) else float(pred_score)
        real_score = float(real_score.detach().cpu()) if torch.is_tensor(real_score) else float(real_score)
        delta_t = real_score - pred_score
        w_t = math.tanh(self.f2p_beta * delta_t)
        trace = {
            "action": action, "action_sha256": hashlib.sha256(action.encode()).hexdigest(),
            "action_token_count": len(self.tlm.tokenizer(action, add_special_tokens=False)["input_ids"]),
            "action_prompt": self._last_action_prompt,
            "previous_observation": prev,
            "real_observation": nxt,
            "action_prior_logprob": prior, "predicted_outcome": self._last_prediction,
            "predicted_outcome_chars": len(self._last_prediction), "predicted_context_action_logprob": pred_score,
            "real_outcome_structured": delta, "real_outcome_text": real_outcome,
            "real_context_action_logprob": real_score, "delta_real_minus_pred": delta_t,
            "beta": self.f2p_beta, "w_t": w_t, "loss_definition": "-w_t * mean_logprob(action|H)",
            "grad_weight_detached": True, "prediction_prompt_sha256": hashlib.sha256(self._last_action_prompt.encode()).hexdigest(),
            "memory_size": len(self.short_term_memory), "training_update": None,
        }
        self.f2p_buffer.append({"action": action, "w_t": w_t, "prompt": self._last_action_prompt})
        if len(self.f2p_buffer) >= self.f2p_update_frequency:
            trace["training_update"] = self._policy_update(self.f2p_buffer)
            self.f2p_buffer = []
        verified = (
            "[Verified transition]\n"
            "Observation before action:\n" + prev
            + "\n"
            "Action actually taken:\n" + action
            + "\nActual environment change:\n" + nxt
            + "\nStructured feedback:\n" + real_outcome
        )
        self.short_term_memory.append(verified)
        self.short_term_memory = self.short_term_memory[-self.short_term_memory_size:]
        self.f2p_last_trace = trace
        self._append_trace(trace)

    def get_f2p_trace(self):
        return self.f2p_last_trace


@lru_cache(maxsize=None)
def create_feedback_to_policy_ttt_agent(Agent: Type):
    class_name = f"FeedbackToPolicyTTTAgent__{Agent.__module__}.{Agent.__name__}"
    return type(class_name, (FeedbackToPolicyTTTAgent, Agent), {"__module__": Agent.__module__, "__agent__": Agent})
