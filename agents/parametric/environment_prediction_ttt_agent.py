from __future__ import annotations

from typing import Any, Dict, List, Optional, Type
from functools import lru_cache
import json

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from agents.parametric.lora_sft_agent import LoRASFTAgent


class _TransitionDataset(Dataset):
    """Causal-LM pairs with loss only on the observed next-state target."""
    def __init__(self, tokenizer, pairs: List[Dict[str, str]], max_len: int):
        self.tokenizer = tokenizer
        self.pairs = pairs
        self.max_len = max_len
        self.pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        prompt_ids = self.tokenizer(
            pair["prompt"], add_special_tokens=True, truncation=False,
        )["input_ids"]
        target_ids = self.tokenizer(
            pair["target"], add_special_tokens=False, truncation=False,
        )["input_ids"]
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            target_ids = target_ids + [eos]
        keep_target = min(len(target_ids), max(1, self.max_len // 2))
        target_ids = target_ids[:keep_target]
        max_prompt = max(1, self.max_len - len(target_ids))
        prompt_ids = prompt_ids[-max_prompt:]
        ids = prompt_ids + target_ids
        labels = [-100] * len(prompt_ids) + target_ids
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones(len(ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def _collate(batch, pad_id):
    return {
        "input_ids": torch.nn.utils.rnn.pad_sequence(
            [x["input_ids"] for x in batch], batch_first=True, padding_value=pad_id
        ),
        "attention_mask": torch.nn.utils.rnn.pad_sequence(
            [x["attention_mask"] for x in batch], batch_first=True, padding_value=0
        ),
        "labels": torch.nn.utils.rnn.pad_sequence(
            [x["labels"] for x in batch], batch_first=True, padding_value=-100
        ),
    }


class EnvironmentPredictionTTTAgent(LoRASFTAgent):
    """LoRA TTT using only (history, action, observed next observation).

    The agent predicts a next environment change in its response, but the only
    training target is the real next observation returned by the environment.
    No reward, Q-value, candidate-action reranking, temporary adapter, or KL
    distillation is used.
    """

    def __init__(self, id: str, name: str, cfg=None, train_cfg=None):
        super().__init__(id=id, name=name, cfg=cfg, train_cfg=train_cfg)
        self.short_term_memory: List[str] = []
        self.transition_pairs: List[Dict[str, str]] = []
        self.memories_since_train = 0
        self._last_prompt = ""
        self._last_prediction = ""
        self._prediction_system = (
            self.cfg.system_prompt
            + "\nIn addition to the required reasoning and action keys, include "
              "a short predicted_outcome key describing the immediate environment change."
        )

    def _act(self, obs: Dict[str, Any]):
        obs_text = obs.get("text", "")
        memory_text = "\n\n".join(self.short_term_memory)
        user_prompt = (
            "My Current Observation:\n"
            + obs_text
            + ("\n\nVerified recent transitions:\n" + memory_text if memory_text else "")
            + "\n\nChoose one valid action and briefly predict its immediate environment change."
        )
        self._last_prompt = user_prompt
        lm_output = self.tlm.generate(
            user_prompt=user_prompt,
            system_prompt=self._prediction_system,
        )
        parsed = self.cfg.json_parser(lm_output["response"])
        if not isinstance(parsed, dict):
            parsed = {"reasoning": "Failed to parse JSON.", "action": "wait", "predicted_outcome": ""}
        action = parsed.get("action", "wait")
        if not isinstance(action, str) or not action.strip():
            action = "wait"
        self._last_prediction = str(parsed.get("predicted_outcome", ""))
        return action.strip(), lm_output["num_input_tokens"], lm_output["num_output_tokens"], lm_output["response"]

    def observe_transition(self, previous_obs, action: str, next_obs, reward=None, info=None):
        """Receive the only supervision available: the real next observation."""
        prev_text = previous_obs.get("text", "") if isinstance(previous_obs, dict) else str(previous_obs)
        next_text = next_obs.get("text", "") if isinstance(next_obs, dict) else str(next_obs)
        verified = (
            "[Verified transition]\n"
            f"Observation before action:\n{prev_text}\n"
            f"Action actually taken:\n{action}\n"
            f"Actual environment change:\n{next_text}"
        )
        self.short_term_memory.append(verified)
        self.short_term_memory = self.short_term_memory[-self.short_term_memory_size:]
        self.transition_pairs.append({
            "prompt": self._last_prompt + "\n\nAction actually taken:\n" + action + "\n\nActual environment change:\n",
            "target": next_text,
        })
        self.transition_pairs = self.transition_pairs[-self.short_term_memory_size:]
        self.memories_since_train += 1
        if len(self.transition_pairs) >= self.short_term_memory_size and self.memories_since_train >= self.short_term_memory_size:
            self._train_on_observed_transitions(list(self.transition_pairs))
            self.memories_since_train = 0

    def _train_on_observed_transitions(self, pairs: List[Dict[str, str]]):
        if not pairs:
            return
        peft_m = self.tlm._ensure_peft()
        peft_m.train()
        ds = _TransitionDataset(self.tlm.tokenizer, pairs, self.max_seq_len)
        dl = DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=lambda b: _collate(b, self.tlm.tokenizer.pad_token_id or self.tlm.tokenizer.eos_token_id),
        )
        opt = AdamW(peft_m.parameters(), lr=self.lr)
        scaler = torch.amp.GradScaler("cuda", enabled=(self.fp16 and torch.cuda.is_available()))
        steps = 0
        losses = []
        for _ in range(self.epochs):
            for batch in dl:
                for key in batch:
                    batch[key] = batch[key].to(self.tlm.device)
                with torch.amp.autocast("cuda", enabled=(self.fp16 and torch.cuda.is_available())):
                    out = peft_m(**batch)
                    loss = out.loss / self.grad_accum
                scaler.scale(loss).backward()
                losses.append(float(loss.detach().cpu()))
                if (steps + 1) % self.grad_accum == 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(peft_m.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                steps += 1
        peft_m.eval()
        self.steps_trained_total += steps
        print(
            f"[EnvironmentPredictionTTT] trained on {len(pairs)} observed transitions; "
            f"steps={steps}, target_loss={sum(losses)/max(1,len(losses)):.4f}",
            flush=True,
        )


@lru_cache(maxsize=None)
def create_environment_prediction_ttt_agent(Agent: Type):
    class_name = f"EnvironmentPredictionTTTAgent__{Agent.__module__}.{Agent.__name__}"
    return type(
        class_name,
        (EnvironmentPredictionTTTAgent, Agent),
        {"__module__": Agent.__module__, "__agent__": Agent},
    )
