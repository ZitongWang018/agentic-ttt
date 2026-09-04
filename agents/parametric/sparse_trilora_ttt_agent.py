from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, deque
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Type

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file, save_file
from torch.optim import AdamW

from agents.parametric.feedback_to_policy_ttt_agent import FeedbackToPolicyTTTAgent
from utils import atomic_write


REWARD_KEYS = (
    "quest", "exploration", "craft", "kill", "unique_kill",
    "side_quest", "trade", "death",
)
DEFAULT_DIAGNOSTIC_POINTS = (
    15, 17, 21, 27, 29, 30, 31, 33, 35, 36, 40, 41, 54, 62, 63,
    65, 76, 80, 86, 94, 97, 100, 106, 121, 126, 129, 142, 165,
    195, 246, 298, 374, 407, 453, 464, 476, 486,
)


def _dist_ready() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _reasoning_prefix(response: str) -> str:
    matches = list(re.finditer(r'"action"\s*:\s*"', response or ""))
    if matches:
        return response[: matches[-1].end()]
    match = re.search(r'"reasoning"\s*:\s*"([\s\S]*?)"', response or "")
    reasoning = match.group(1) if match else ""
    return json.dumps({"reasoning": reasoning}, ensure_ascii=False)[:-1] + ', "action": "'


def _reward_dict(reward: Any) -> Dict[str, float]:
    if reward is None:
        return {}
    if isinstance(reward, dict):
        return reward
    return dict(getattr(reward, "__dict__", {}) or {})


def _reward_scalar(reward: Dict[str, Any]) -> float:
    return sum(float(reward.get(k, 0) or 0) for k in REWARD_KEYS if k != "death") - float(
        reward.get("death", 0) or 0
    )


class SparseTriLoRATTTAgent(FeedbackToPolicyTTTAgent):
    """Online Task-r12 + Top-1 Free-r4/r4 TTT with two-GPU data parallel updates.

    Rank 0 owns the environment.  All ranks own synchronized model replicas.
    Actual decisions use Task + exactly one Free adapter for a fixed block.
    Task batches and Free blocks are sharded across ranks; gradients are summed
    before either optimizer steps.  At selected steps, six counterfactual paths
    are generated in parallel, but only the routed Task+Free path is executed.
    """

    def __init__(self, id: str, name: str, cfg=None, train_cfg=None):
        super().__init__(id=id, name=name, cfg=cfg, train_cfg=train_cfg)
        self.task_rank = int(getattr(self.cfg, "task_rank", 12))
        self.free_rank = int(getattr(self.cfg, "free_rank", 4))
        self.free_scale = float(getattr(self.cfg, "free_scale", 0.25))
        self.block_horizon = int(getattr(self.cfg, "free_block_horizon", 10))
        self.gamma = float(getattr(self.cfg, "free_gamma", 0.99))
        self.free_lr = float(getattr(self.cfg, "free_lr", self.lr))
        self.kl_coef = float(getattr(self.cfg, "free_kl_coef", 0.05))
        self.sep_coef = float(getattr(self.cfg, "free_sep_coef", 0.01))
        self.sep_margin = float(getattr(self.cfg, "free_sep_margin", 0.0))
        configured_points = str(getattr(self.cfg, "trilora_diagnostic_points", "") or "")
        self.diagnostic_points = (
            {int(x.strip()) for x in configured_points.split(",") if x.strip()}
            if configured_points else set(DEFAULT_DIAGNOSTIC_POINTS)
        )
        self._init_adapters()
        self.task_opt = AdamW(self._adapter_parameters("task"), lr=self.lr)
        self.free_opts = {
            name: AdamW(self._adapter_parameters(name), lr=self.free_lr)
            for name in ("free1", "free2")
        }
        self.f2p_buffer = []
        self.free_block: List[Dict[str, Any]] = []
        self.block_returns: List[float] = []
        self.route_counts = Counter()
        self.online_step = 0
        self.active_expert = "free1"
        self.trilora_last_trace: Dict[str, Any] = {}
        self.trilora_log_path: str | None = None
        self.free_data_paths: Dict[str, str] = {}
        self.diagnostic_log_path: str | None = None
        self._last_response = ""
        self._recent_actions = deque(maxlen=25)

    def _init_adapters(self):
        target = ["q_proj", "k_proj", "v_proj", "o_proj"]
        task_cfg = LoraConfig(
            r=self.task_rank, lora_alpha=self.task_rank, lora_dropout=0.0,
            target_modules=target, bias="none", task_type="CAUSAL_LM",
        )
        free_cfg = LoraConfig(
            r=self.free_rank, lora_alpha=self.free_rank, lora_dropout=0.0,
            target_modules=target, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(self.tlm.model, task_cfg, adapter_name="task")
        model.add_adapter("free1", free_cfg)
        model.add_adapter("free2", free_cfg)
        self.tlm._peft = model
        self.tlm.model = model
        self.tlm.base.model = model
        self._set_adapters(["task", "free1"], ())

    def _adapter_parameters(self, adapter: str):
        return [p for n, p in self.tlm.model.named_parameters() if f".{adapter}." in n]

    def _set_adapters(self, names: List[str], trainable=()):
        model = self.tlm.model
        model.base_model.set_adapter(names if len(names) > 1 else names[0])
        trainable = set(trainable)
        for name, param in model.named_parameters():
            param.requires_grad = any(f".{adapter}." in name for adapter in trainable)
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.set_scale("task", 1.0)
                module.set_scale("free1", self.free_scale)
                module.set_scale("free2", self.free_scale)

    def _parse_output(self, lm_output):
        response = (lm_output or {}).get("response") or ""
        parsed = self.cfg.json_parser(response)
        if not isinstance(parsed, dict):
            parsed = {"reasoning": "parse failure", "action": "wait", "predicted_outcome": ""}
        action = parsed.get("action", "wait")
        if not isinstance(action, str) or not action.strip():
            action = "wait"
        return {
            "reasoning": str(parsed.get("reasoning", "")),
            "action": action.strip(),
            "predicted_outcome": str(parsed.get("predicted_outcome", "")),
            "raw": response,
            "num_input_tokens": int((lm_output or {}).get("num_input_tokens", 0)),
            "num_output_tokens": int((lm_output or {}).get("num_output_tokens", 0)),
        }

    def _generate_path(self, prompt: str, adapters: List[str]):
        model = self.tlm.model
        model.eval()
        if adapters:
            self._set_adapters(adapters, ())
            output = self.tlm.generate(user_prompt=prompt, system_prompt=self.f2p_system)
        else:
            with model.disable_adapter():
                output = self.tlm.generate(user_prompt=prompt, system_prompt=self.f2p_system)
        return self._parse_output(output)

    def _local_diagnostics(self, prompt: str, step: int, paths: Dict[str, List[str]]):
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state_all()
        result = {}
        try:
            for name, adapters in paths.items():
                seed = int(getattr(self.cfg, "torch_seed", 42)) + step * 1000
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                result[name] = self._generate_path(prompt, adapters)
        finally:
            torch.random.set_rng_state(cpu_state)
            torch.cuda.set_rng_state_all(cuda_state)
        return result

    def _diagnostic_generate(self, prompt: str, step: int):
        rank0_paths = {
            "base": [], "task_free1": ["task", "free1"],
            "task_free2": ["task", "free2"],
        }
        rank1_paths = {
            "task": ["task"], "free1_only": ["free1"], "free2_only": ["free2"],
        }
        if _dist_ready():
            self._broadcast({"kind": "diagnostic", "prompt": prompt, "step": step})
            local = self._local_diagnostics(prompt, step, rank0_paths)
            gathered = [None for _ in range(dist.get_world_size())]
            dist.gather_object(local, gathered, dst=0)
            merged = {}
            for part in gathered:
                merged.update(part or {})
            return merged
        merged = self._local_diagnostics(prompt, step, rank0_paths)
        merged.update(self._local_diagnostics(prompt, step, rank1_paths))
        return merged

    def _act(self, obs: Dict[str, Any]):
        obs_text = obs.get("text", "")
        memory_text = "\n\n".join(self.short_term_memory)
        self._last_action_prompt = (
            "My Current Observation:\n" + obs_text
            + ("\n\nVerified recent feedback:\n" + memory_text if memory_text else "")
            + "\n\nChoose one valid action and briefly predict its immediate environment change."
        )
        if self.online_step in self.diagnostic_points:
            diagnostics = self._diagnostic_generate(self._last_action_prompt, self.online_step)
            chosen = diagnostics[f"task_{self.active_expert}"]
            self._append_jsonl(self.diagnostic_log_path, {
                "step": self.online_step, "active_expert": self.active_expert,
                "executed_path": f"task_{self.active_expert}", "decisions": diagnostics,
            })
        else:
            diagnostics = None
            chosen = self._generate_path(
                self._last_action_prompt, ["task", self.active_expert]
            )
        self._last_action = chosen["action"]
        self._last_prediction = chosen["predicted_outcome"]
        self._last_response = chosen["raw"]
        self._set_adapters(["task"], ())
        prior = self._score_action("", self._last_action, requires_grad=False)
        self._last_action_prior = float(prior.detach().cpu())
        self.trilora_last_trace = {
            "online_step": self.online_step,
            "active_expert": 1 if self.active_expert == "free1" else 2,
            "diagnostic_point": int(diagnostics is not None),
            "diagnostic_unique_actions": len({x["action"] for x in diagnostics.values()}) if diagnostics else 0,
        }
        return (
            self._last_action, chosen["num_input_tokens"],
            chosen["num_output_tokens"], chosen["raw"],
        )

    def _action_logits(self, prompt, action, prefix="", requires_grad=False):
        tokenizer = self.tlm.tokenizer
        rendered = tokenizer.apply_chat_template(
            [{"role": "system", "content": self.f2p_system}, {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        conditioned = rendered + prefix
        prompt_ids = tokenizer(conditioned, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(conditioned + action, add_special_tokens=False)["input_ids"]
        start = max(0, len(full_ids) - self.f2p_max_score_len)
        ids = torch.tensor(full_ids[start:], dtype=torch.long, device=self.tlm.device).unsqueeze(0)
        first = min(max(0, len(prompt_ids) - start - 1), ids.shape[1] - 1)
        count = ids.shape[1] - 1 - first
        if count <= 0:
            raise ValueError(f"No scoreable action tokens: {action!r}")
        with torch.set_grad_enabled(requires_grad):
            logits = self.tlm.model(
                input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=count + 1
            ).logits[0, :-1].float()
        return logits, ids[0, -count:]

    @staticmethod
    def _mean_logp(logits, labels):
        return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).mean()

    def _allreduce_grads(self, adapter: str):
        params = self._adapter_parameters(adapter)
        if not _dist_ready():
            return params
        for param in params:
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        return params

    def _reduce_stats(self, values: List[float]):
        tensor = torch.tensor(values, dtype=torch.float64, device=self.tlm.device)
        if _dist_ready():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.cpu().tolist()

    def _task_update_local(self, batch):
        self._set_adapters(["task"], ["task"])
        self.tlm.model.train()
        self.task_opt.zero_grad(set_to_none=True)
        rank = dist.get_rank() if _dist_ready() else 0
        world = dist.get_world_size() if _dist_ready() else 1
        loss_sum = weight_sum = 0.0
        for item in batch[rank::world]:
            logits, labels = self._action_logits(item["prompt"], item["action"], requires_grad=True)
            loss = -float(item["w_t"]) * self._mean_logp(logits, labels)
            (loss / len(batch)).backward()
            loss_sum += float(loss.detach().cpu())
            weight_sum += float(item["w_t"])
        params = self._allreduce_grads("task")
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        self.task_opt.step()
        stats = self._reduce_stats([loss_sum, weight_sum])
        self.tlm.model.eval()
        return {
            "updated": True, "loss": stats[0] / len(batch),
            "w_mean": stats[1] / len(batch), "grad_norm": float(grad),
            "batch_size": len(batch), "distributed_world_size": world,
        }

    def _policy_update(self, batch):
        if not batch:
            return {"updated": False}
        if _dist_ready():
            self._broadcast({"kind": "task_update", "batch": batch})
        result = self._task_update_local(batch)
        self.steps_trained_total += 1
        return result

    def _free_update_local(self, block, expert, advantage, block_return):
        rank = dist.get_rank() if _dist_ready() else 0
        world = dist.get_world_size() if _dist_ready() else 1
        opt = self.free_opts[expert]
        opt.zero_grad(set_to_none=True)
        loss_sum = kl_sum = 0.0
        for row in block[rank::world]:
            prompt, action, prefix = row["prompt"], row["action"], row["reasoning_prefix"]
            self._set_adapters(["task"], ())
            with torch.no_grad():
                task_logits, labels = self._action_logits(prompt, action, prefix, False)
                task_logdist = F.log_softmax(task_logits, dim=-1)
            self._set_adapters(["task", expert], [expert])
            combined_logits, labels = self._action_logits(prompt, action, prefix, True)
            e2e = -advantage * self._mean_logp(combined_logits, labels)
            combined_logdist = F.log_softmax(combined_logits, dim=-1)
            kl = (task_logdist.exp() * (task_logdist - combined_logdist)).sum(-1).mean()
            ((e2e + self.kl_coef * kl) / len(block)).backward()
            loss_sum += float(e2e.detach().cpu())
            kl_sum += float(kl.detach().cpu())

        anchor = block[-1]
        other = "free2" if expert == "free1" else "free1"
        self._set_adapters(["task"], ())
        with torch.no_grad():
            task_anchor, _ = self._action_logits(
                anchor["prompt"], anchor["action"], anchor["reasoning_prefix"], False
            )
            self._set_adapters(["task", other], ())
            other_logits, _ = self._action_logits(
                anchor["prompt"], anchor["action"], anchor["reasoning_prefix"], False
            )
            other_delta = other_logits - task_anchor
        self._set_adapters(["task", expert], [expert])
        active_logits, _ = self._action_logits(
            anchor["prompt"], anchor["action"], anchor["reasoning_prefix"], True
        )
        cosine = F.cosine_similarity(
            (active_logits - task_anchor).reshape(1, -1), other_delta.reshape(1, -1)
        ).mean()
        separation = F.relu(cosine - self.sep_margin)
        (self.sep_coef * separation / world).backward()
        params = self._allreduce_grads(expert)
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        stats = self._reduce_stats([loss_sum, kl_sum])
        return {
            "block_return": block_return, "block_advantage": advantage,
            "e2e_loss": stats[0] / len(block), "kl_loss": stats[1] / len(block),
            "separation_loss": float(separation.detach().cpu()),
            "delta_cosine": float(cosine.detach().cpu()), "free_grad_norm": float(grad),
            "distributed_world_size": world,
        }

    def _train_free_block(self, block, expert):
        block_return = sum((self.gamma ** i) * _reward_scalar(row["reward"]) for i, row in enumerate(block))
        baseline = sum(self.block_returns) / len(self.block_returns) if self.block_returns else 0.0
        if len(self.block_returns) >= 2:
            var = sum((x - baseline) ** 2 for x in self.block_returns) / len(self.block_returns)
            advantage = (block_return - baseline) / max(math.sqrt(var), 1.0)
        else:
            advantage = block_return - baseline
        advantage = max(-3.0, min(3.0, advantage))
        if _dist_ready():
            self._broadcast({
                "kind": "free_update", "block": block, "expert": expert,
                "advantage": advantage, "block_return": block_return,
            })
        metrics = self._free_update_local(block, expert, advantage, block_return)
        self.block_returns.append(block_return)
        self.route_counts[expert] += 1
        for row in block:
            self._append_jsonl(self.free_data_paths.get(expert), {
                "expert": expert, "block_id": sum(self.route_counts.values()) - 1,
                "environment_step": row["step"],
                "x": {"action_prompt": row["prompt"], "realized_reasoning_prefix": row["reasoning_prefix"]},
                "y": row["action"], "advantage": advantage,
                "block_return": block_return, "source": "online_environment",
                "counterfactual_probe": False,
            })
        return metrics

    @staticmethod
    def _append_jsonl(path, record):
        if not path:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _broadcast(self, command):
        payload = [command]
        dist.broadcast_object_list(payload, src=0)

    def distributed_worker_loop(self):
        if not _dist_ready() or dist.get_rank() == 0:
            return
        while True:
            payload = [None]
            dist.broadcast_object_list(payload, src=0)
            command = payload[0]
            kind = command.get("kind")
            if kind == "stop":
                break
            if kind == "task_update":
                self._task_update_local(command["batch"])
            elif kind == "free_update":
                self._free_update_local(
                    command["block"], command["expert"],
                    command["advantage"], command["block_return"],
                )
            elif kind == "diagnostic":
                paths = {"task": ["task"], "free1_only": ["free1"], "free2_only": ["free2"]}
                local = self._local_diagnostics(command["prompt"], command["step"], paths)
                dist.gather_object(local, None, dst=0)
            else:
                raise RuntimeError(f"Unknown distributed command: {kind}")

    def stop_distributed_workers(self):
        if _dist_ready() and dist.get_rank() == 0:
            self._broadcast({"kind": "stop"})

    def observe_transition(self, previous_obs, action: str, next_obs, reward=None, info=None):
        prev = previous_obs.get("text", "") if isinstance(previous_obs, dict) else str(previous_obs)
        nxt = next_obs.get("text", "") if isinstance(next_obs, dict) else str(next_obs)
        reward_data = _reward_dict(reward)
        delta = self._state_delta(prev, nxt, action=action)
        real_outcome = self._outcome_text(delta)
        self._set_adapters(["task"], ())
        pred_score = float(self._score_action(self._last_prediction, action, False).detach().cpu())
        real_score = float(self._score_action(real_outcome, action, False).detach().cpu())
        prior = float(self._last_action_prior)
        delta_t = real_score - pred_score
        w_t = math.tanh(self.f2p_beta * delta_t)
        self.f2p_buffer.append({"action": action, "w_t": w_t, "prompt": self._last_action_prompt})
        task_update = None
        if len(self.f2p_buffer) >= self.f2p_update_frequency:
            task_update = self._policy_update(self.f2p_buffer)
            self.f2p_buffer = []

        self.free_block.append({
            "step": self.online_step, "prompt": self._last_action_prompt,
            "action": action, "reasoning_prefix": _reasoning_prefix(self._last_response),
            "response": self._last_response, "reward": reward_data,
        })
        free_update = None
        if len(self.free_block) >= self.block_horizon:
            free_update = self._train_free_block(self.free_block, self.active_expert)
            self.free_block = []
            self.active_expert = "free1" if self.active_expert == "free2" else "free2"

        verified = (
            "[Verified transition]\nObservation before action:\n" + prev
            + "\nAction actually taken:\n" + action
            + "\nActual environment change:\n" + nxt
            + "\nStructured feedback:\n" + real_outcome
        )
        self.short_term_memory.append(verified)
        self.short_term_memory = self.short_term_memory[-self.short_term_memory_size:]
        self._recent_actions.append(action)
        self.f2p_last_trace = {
            "action": action, "action_prompt": self._last_action_prompt,
            "action_prior_logprob": prior, "predicted_outcome": self._last_prediction,
            "predicted_context_action_logprob": pred_score,
            "real_context_action_logprob": real_score,
            "delta_real_minus_pred": delta_t, "w_t": w_t,
            "loss_definition": "-w_t * mean_logprob(action|state)",
            "training_update": task_update,
        }
        self.trilora_last_trace.update({
            "task_update": task_update or {}, "free_update": free_update or {},
            "active_rank": self.task_rank + self.free_rank,
            "stored_rank": self.task_rank + 2 * self.free_rank,
            "free1_blocks": self.route_counts["free1"],
            "free2_blocks": self.route_counts["free2"],
        })
        self._append_trace(self.f2p_last_trace)
        self._append_jsonl(self.trilora_log_path, self.trilora_last_trace)
        self.online_step += 1

    def get_trilora_trace(self):
        return self.trilora_last_trace

    def finish_episode(self):
        if self.f2p_buffer:
            self.f2p_last_trace["episode_end_training_update"] = self._policy_update(self.f2p_buffer)
            self.f2p_buffer = []
        if self.free_block:
            self.trilora_last_trace["episode_end_free_update"] = self._train_free_block(
                self.free_block, self.active_expert
            )
            self.free_block = []

    def _save_adapter_exact(self, root: Path, adapter: str):
        target = root / adapter
        target.mkdir(parents=True, exist_ok=True)
        self.tlm.model.peft_config[adapter].save_pretrained(target)
        state = get_peft_model_state_dict(self.tlm.model, adapter_name=adapter)
        state = {k: v.detach().cpu().contiguous() for k, v in state.items()}
        if not state:
            raise RuntimeError(f"Adapter {adapter} produced an empty state dict")
        save_file(state, str(target / "adapter_model.safetensors"))

    def save_memory(self, full_memory_dir: str) -> None:
        root = Path(full_memory_dir)
        adapter_root = root / "lora"
        for adapter in ("task", "free1", "free2"):
            self._save_adapter_exact(adapter_root, adapter)
        torch.save({
            "task": self.task_opt.state_dict(),
            "free1": self.free_opts["free1"].state_dict(),
            "free2": self.free_opts["free2"].state_dict(),
        }, root / "optimizer_states.pt")
        state = {
            "agent_id": self.id, "agent_name": self.name,
            "algorithm": "online_sparse_trilora_rankmatched_v2",
            "online_step": self.online_step, "active_expert": self.active_expert,
            "route_counts": dict(self.route_counts), "block_returns": self.block_returns,
            "f2p_buffer": self.f2p_buffer, "free_block": self.free_block,
            "short_term_memory": self.short_term_memory,
            "task_rank": self.task_rank, "free_rank": self.free_rank,
            "free_scale": self.free_scale, "distributed_world_size": dist.get_world_size() if _dist_ready() else 1,
        }
        atomic_write(str(root / self.memory_paths[0]), json.dumps(state, ensure_ascii=False, indent=2))

    def load_memory(self, full_memory_dir: str) -> None:
        root = Path(full_memory_dir)
        state_path = root / self.memory_paths[0]
        if not state_path.exists():
            return
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("algorithm") != "online_sparse_trilora_rankmatched_v2":
            raise ValueError(f"Incompatible SparseTriLoRA checkpoint: {state_path}")
        for adapter in ("task", "free1", "free2"):
            weight_path = root / "lora" / adapter / "adapter_model.safetensors"
            if not weight_path.exists():
                raise FileNotFoundError(f"Missing adapter checkpoint: {weight_path}")
            adapter_state = load_file(str(weight_path), device=str(self.tlm.device))
            set_peft_model_state_dict(self.tlm.model, adapter_state, adapter_name=adapter)
        optimizer_path = root / "optimizer_states.pt"
        if optimizer_path.exists():
            optimizer_states = torch.load(optimizer_path, map_location=self.tlm.device, weights_only=False)
            self.task_opt.load_state_dict(optimizer_states["task"])
            self.free_opts["free1"].load_state_dict(optimizer_states["free1"])
            self.free_opts["free2"].load_state_dict(optimizer_states["free2"])
        self.online_step = int(state.get("online_step", 0))
        self.active_expert = str(state.get("active_expert", "free1"))
        self.route_counts = Counter(state.get("route_counts", {}) or {})
        self.block_returns = list(state.get("block_returns", []) or [])
        self.f2p_buffer = list(state.get("f2p_buffer", []) or [])
        self.free_block = list(state.get("free_block", []) or [])
        self.short_term_memory = list(state.get("short_term_memory", []) or [])


@lru_cache(maxsize=None)
def create_sparse_trilora_ttt_agent(Agent: Type):
    class_name = f"SparseTriLoRATTTAgent__{Agent.__module__}.{Agent.__name__}"
    return type(class_name, (SparseTriLoRATTTAgent, Agent), {"__module__": Agent.__module__, "__agent__": Agent})
