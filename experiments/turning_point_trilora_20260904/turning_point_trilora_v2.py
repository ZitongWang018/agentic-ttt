"""Rank-matched causal replay for Task + two sparse Free LoRA experts.

The environment is never replayed.  A fresh set of three adapters is trained
once, in trajectory order, from the historical F2P JSONL.  At a turning point,
generation happens before that row is added to training, so the probe only uses
information that was available strictly before the decision.  Six diagnostic
paths are generated at each point, but none of those counterfactual actions is
ever inserted into training.  Free experts only consume their assigned causal
trajectory blocks, stored explicitly as (x, y, A_b) records.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, deque
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from torch.optim import AdamW
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_POINTS = (
    15, 17, 21, 27, 29, 30, 31, 33, 35, 36, 40, 41, 54, 62, 63, 65,
    76, 80, 86, 94, 97, 100, 106, 121, 126, 129, 142, 165, 195, 246,
    298, 374, 407, 453, 464, 476, 486,
)
REWARD_KEYS = (
    "quest", "exploration", "craft", "kill", "unique_kill",
    "side_quest", "trade", "death",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-log", required=True)
    p.add_argument("--model", default="/root/autodl-tmp/model/Qwen3-4B")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--name", default="remnant-trilora-turning-point-replay")
    p.add_argument("--swanlab-group", default="remnant-point-test")
    p.add_argument("--points", default=",".join(map(str, DEFAULT_POINTS)))
    p.add_argument("--max-replay-step", type=int, default=499)
    p.add_argument("--block-size", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.97)
    p.add_argument("--task-lr", type=float, default=5e-6)
    p.add_argument("--free-lr", type=float, default=2e-5)
    p.add_argument("--task-update-frequency", type=int, default=5)
    p.add_argument("--task-rank", type=int, default=12)
    p.add_argument("--free-rank", type=int, default=4)
    p.add_argument("--baseline-active-rank", type=int, default=16)
    p.add_argument("--free-scale", type=float, default=0.25)
    p.add_argument("--sep-coef", type=float, default=0.01)
    p.add_argument("--sep-margin", type=float, default=0.0)
    p.add_argument("--kl-coef", type=float, default=0.05)
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--repetition-penalty", type=float, default=1.3)
    p.add_argument("--device-map", choices=["single", "balanced"], default="single")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--disable-swanlab", action="store_true")
    return p.parse_args()


class SwanSidecar:
    def __init__(self, output_dir: Path, name: str, group: str, config: dict, enabled: bool):
        self.proc = None
        self.handle = None
        if not enabled:
            return
        env = os.environ.copy()
        for key in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env.pop(key, None)
        sidecar = REPO_ROOT / "tools" / "swanlab_sidecar.py"
        py = Path("/root/autodl-tmp/swanlab-venv/bin/python")
        self.handle = (output_dir / "swanlab-sidecar.log").open("a", encoding="utf-8", buffering=1)
        self.proc = subprocess.Popen(
            [str(py), str(sidecar)], stdin=subprocess.PIPE, stdout=self.handle,
            stderr=subprocess.STDOUT, text=True, bufsize=1, env=env,
        )
        self.send({"type": "init", "kwargs": {
            "project": "agentic-TTT", "workspace": "ZitongWang",
            "name": name, "group": group, "job_type": "turning-point-eval",
            "tags": ["remnant", "trilora", "turning-point", "offline-replay"],
            "config": config, "log_dir": str(output_dir / "swanlog"), "mode": "online",
        }, "run_id_path": str(output_dir / ".swanlab_run_id")})

    def send(self, message: dict):
        if self.proc is None or self.proc.stdin is None:
            return
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def log(
        self, step: int, metrics: dict, text_payload: dict | None = None,
        text_key: str = "eval/decision_context", caption: str | None = None,
    ):
        msg = {"type": "log", "step": step, "metrics": metrics}
        if text_payload is not None:
            msg["texts"] = {text_key: {
                "data": json.dumps(text_payload, ensure_ascii=False, indent=2),
                "caption": caption or f"Turning-point decision at historical step {step}",
            }}
        self.send(msg)

    def finish(self):
        if self.proc is None:
            return
        try:
            self.send({"type": "finish"})
            self.proc.stdin.close()
            self.proc.wait(timeout=120)
        finally:
            if self.handle:
                self.handle.close()


def build_system_prompt():
    from games.generated.remnant.agent import Agent
    actions = Agent("probe", "probe").available_actions
    action_text = "\n".join(
        f"- {a.verb} " + " ".join(f"<{x}>" for x in a.params) for a in actions
    )
    system = f"""
You are the player in a text adventure games. The world is described in text form.
At each turn, you may choose ONE action from the action space below.

Action space:
{action_text}

Output format (STRICT):
Return a single JSON object with exactly these keys:
{{
  "reasoning": "A few sentences explaining why you choose the action.",
  "action": "<action>"
}}

Rules:
- The JSON must be the ONLY content in your reply (no extra text before/after).
- The action must exactly match one option from the action space.

F2P OUTPUT FORMAT OVERRIDE:
For this agent, the earlier two-key output schema is replaced by this exact three-key JSON schema:
{{
  "reasoning": "A few sentences explaining why you choose the action.",
  "action": "<action>",
  "predicted_outcome": "A short prediction of the immediate environment change."
}}
All other action-validity and JSON-only rules above still apply.
""".strip()
    return system, {a.verb for a in actions}


def set_adapters(model, names, trainable=(), free_scale=0.25):
    if not names:
        raise ValueError("Use model.disable_adapter() for the base-only path")
    model.base_model.set_adapter(list(names) if len(names) > 1 else names[0])
    trainable = set(trainable)
    for name, param in model.named_parameters():
        param.requires_grad = any(f".{adapter}." in name for adapter in trainable)
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.set_scale("task", 1.0)
            module.set_scale("free1", free_scale)
            module.set_scale("free2", free_scale)


def action_logits(
    model, tokenizer, system, prompt, action, max_seq_len, requires_grad,
    assistant_prefix="",
):
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    conditioned = rendered + assistant_prefix
    prompt_ids = tokenizer(conditioned, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(conditioned + action, add_special_tokens=False)["input_ids"]
    action_start = len(prompt_ids)
    start = max(0, len(full_ids) - max_seq_len)
    ids = torch.tensor(full_ids[start:], dtype=torch.long, device=model.device).unsqueeze(0)
    first = max(0, action_start - start - 1)
    first = min(first, ids.shape[1] - 1)
    count = ids.shape[1] - 1 - first
    if count <= 0:
        raise ValueError(f"Action has no scoreable tokens: {action!r}")
    with torch.set_grad_enabled(requires_grad):
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=count + 1)
        logits = out.logits[0, :-1].float()
    labels = ids[0, -count:]
    return logits, labels


def mean_action_logp(logits, labels):
    return F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).mean()


def reward_scalar(row):
    r = row.get("reward") or {}
    return sum(float(r.get(k, 0) or 0) for k in REWARD_KEYS if k != "death") - float(r.get("death", 0) or 0)


def benchmark_metrics(row, cumulative):
    reward = row.get("reward") or {}
    metrics = {}
    for key in REWARD_KEYS:
        value = float(reward.get(key, 0) or 0)
        cumulative[key] += value
        metrics[f"benchmark/{key}"] = value
        metrics[f"benchmark/cumulative_{key}"] = cumulative[key]
    score = reward_scalar(row)
    cumulative["score"] += score
    metrics.update({
        "benchmark/step_score": score,
        "benchmark/cumulative_score": cumulative["score"],
        "agent/invalid_action": float(bool(row.get("invalid_action", False))),
        "agent/decision_time": float(row.get("decision_time", 0) or 0),
        "agent/input_tokens": float(row.get("num_input_tokens", 0) or 0),
        "agent/output_tokens": float(row.get("num_output_tokens", 0) or 0),
    })
    return metrics


def death_event_payload(row, recent_actions):
    trace = row.get("f2p_trace") or {}
    after = trace.get("real_observation") or (row.get("observation") or {}).get("text", "")
    time_match = re.search(r"Current Time:\s*([^\n]+)", str(after))
    return {
        "event": "death",
        "historical_step": int(row.get("step", -1)),
        "environment_time": time_match.group(1).strip() if time_match else None,
        "recent_actions_before": list(recent_actions),
        "x_before_action": trace.get("action_prompt") or trace.get("previous_observation"),
        "assistant_response": row.get("response"),
        "executed_action": row.get("action"),
        "environment_after_action": after,
        "reward": row.get("reward"),
    }


def parse_response(text):
    match = re.search(r"\{[\s\S]*\}", text or "")
    try:
        obj = json.loads(match.group(0) if match else text)
    except Exception:
        # Qwen occasionally follows the semantic schema but renders labeled
        # prose instead of JSON.  Preserve its actual decision for evaluation
        # rather than incorrectly counting every such answer as ``wait``.
        action_match = re.search(
            r"(?:valid\s+action|action)\s*:\s*[`\"']?([^`\"'\n]+)",
            text or "", flags=re.I,
        )
        outcome_match = re.search(
            r"(?:predicted\s+(?:environment\s+)?(?:change|outcome)|immediate\s+environment\s+change)\s*:\s*([\s\S]+)",
            text or "", flags=re.I,
        )
        if action_match:
            return {
                "reasoning": "non-JSON labeled response",
                "action": action_match.group(1).strip().rstrip(". "),
                "predicted_outcome": outcome_match.group(1).strip() if outcome_match else "",
                "raw": text,
            }
        return {"reasoning": "parse failure", "action": "wait", "predicted_outcome": "", "raw": text}
    return {
        "reasoning": str(obj.get("reasoning", "")),
        "action": str(obj.get("action", "wait")).strip() or "wait",
        "predicted_outcome": str(obj.get("predicted_outcome", "")),
        "raw": text,
    }


def reasoning_prefix(row):
    """Return the realized assistant prefix immediately before the action value.

    The prefix contains the model's sampled thinking/reasoning and JSON field
    prefix.  It is input context (x), never a supervised target.  Using the
    realized prefix keeps Free-LoRA action scoring aligned with inference.
    """
    response = str(row.get("response") or "")
    matches = list(re.finditer(r'"action"\s*:\s*"', response))
    if matches:
        return response[:matches[-1].end()]
    parsed = parse_response(response)
    reasoning = parsed.get("reasoning", "")
    return json.dumps({"reasoning": reasoning}, ensure_ascii=False)[:-1] + ', "action": "'


class JsonActionStop(StoppingCriteria):
    def __init__(self, tokenizer, prompt_length):
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores, **kwargs):
        generated = self.tokenizer.decode(input_ids[0, self.prompt_length:], skip_special_tokens=False)
        return bool(re.search(r'"action"\s*:', generated) and re.search(r"}\s*$", generated))


@torch.inference_mode()
def generate(model, tokenizer, system, prompt, names, args):
    if names:
        set_adapters(model, names, (), args.free_scale)
        adapter_context = nullcontext()
    else:
        for param in model.parameters():
            param.requires_grad = False
        adapter_context = model.disable_adapter()
    model.eval()
    text = tokenizer.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    keep = args.max_seq_len - args.max_new_tokens
    ids = ids[:, -keep:].to(model.device)
    old_use_cache = model.config.use_cache
    model.config.use_cache = True
    try:
        with adapter_context:
            processors = LogitsProcessorList()
            if args.repetition_penalty != 1.0:
                processors.append(RepetitionPenaltyLogitsProcessor(args.repetition_penalty))
            output = model.generate(
                input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=args.max_new_tokens,
                do_sample=True, temperature=args.temperature, top_p=args.top_p,
                logits_processor=processors,
                stopping_criteria=StoppingCriteriaList([JsonActionStop(tokenizer, ids.shape[1])]),
                pad_token_id=tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id,
            )
    finally:
        model.config.use_cache = old_use_cache
    decoded = tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)
    return parse_response(decoded)


def train_task(model, tokenizer, system, batch, optimizer, args):
    set_adapters(model, ["task"], ["task"], args.free_scale)
    model.train(); optimizer.zero_grad(set_to_none=True)
    losses, weights = [], []
    for item in batch:
        logits, labels = action_logits(model, tokenizer, system, item["prompt"], item["action"], args.max_seq_len, True)
        w = float(item["w"])
        loss = -w * mean_action_logp(logits, labels)
        (loss / len(batch)).backward()
        losses.append(float(loss.detach().cpu())); weights.append(w)
    grad = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return {"train/task_loss": sum(losses) / len(losses), "train/task_w_mean": sum(weights) / len(weights), "train/task_grad_norm": float(grad)}


def train_free_block(
    model, tokenizer, system, block, expert, optimizers, returns, route_counts,
    dataset_path, block_id, args,
):
    gamma_return = sum((args.gamma ** i) * reward_scalar(row) for i, row in enumerate(block))
    baseline = sum(returns) / len(returns) if returns else 0.0
    if len(returns) >= 2:
        variance = sum((x - baseline) ** 2 for x in returns) / len(returns)
        advantage = (gamma_return - baseline) / max(math.sqrt(variance), 1.0)
    else:
        advantage = gamma_return - baseline
    advantage = max(-3.0, min(3.0, advantage))
    returns.append(gamma_return); route_counts[expert] += 1
    for opt in optimizers.values(): opt.zero_grad(set_to_none=True)

    training_records = []
    for row in block:
        trace = row.get("f2p_trace") or {}
        prompt = trace.get("action_prompt") or ("My Current Observation:\n" + (row.get("observation") or {}).get("text", ""))
        training_records.append({
            "expert": expert,
            "block_id": block_id,
            "historical_step": int(row.get("step", -1)),
            "x": {"action_prompt": prompt, "realized_reasoning_prefix": reasoning_prefix(row)},
            "y": str(row.get("action", "wait")),
            "advantage": advantage,
            "block_return": gamma_return,
            "source": "historical_f2p_offline_replay",
            "counterfactual_probe": False,
        })
    with dataset_path.open("a", encoding="utf-8") as handle:
        for record in training_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    e2e_values, kl_values = [], []
    for row in block:
        prompt = (row.get("f2p_trace") or {}).get("action_prompt") or ("My Current Observation:\n" + (row.get("observation") or {}).get("text", ""))
        action = str(row.get("action", "wait"))
        prefix = reasoning_prefix(row)
        set_adapters(model, ["task"], [expert], args.free_scale)
        with torch.no_grad():
            task_logits, labels = action_logits(
                model, tokenizer, system, prompt, action, args.max_seq_len, False,
                assistant_prefix=prefix,
            )
            task_logdist = F.log_softmax(task_logits, dim=-1)
        set_adapters(model, ["task", expert], [expert], args.free_scale)
        combined_logits, labels = action_logits(
            model, tokenizer, system, prompt, action, args.max_seq_len, True,
            assistant_prefix=prefix,
        )
        e2e_term = -advantage * mean_action_logp(combined_logits, labels)
        combined_logdist = F.log_softmax(combined_logits, dim=-1)
        kl_term = (task_logdist.exp() * (task_logdist - combined_logdist)).sum(-1).mean()
        ((e2e_term + args.kl_coef * kl_term) / len(block)).backward()
        e2e_values.append(float(e2e_term.detach().cpu()))
        kl_values.append(float(kl_term.detach().cpu()))
        del task_logits, task_logdist, combined_logits, combined_logdist, e2e_term, kl_term

    anchor = block[-1]
    prompt = (anchor.get("f2p_trace") or {}).get("action_prompt") or ("My Current Observation:\n" + (anchor.get("observation") or {}).get("text", ""))
    action = str(anchor.get("action", "wait"))
    prefix = reasoning_prefix(anchor)
    other = "free2" if expert == "free1" else "free1"
    # The inactive expert is a stop-gradient reference.  Only the expert that
    # controlled this block receives a parameter write.
    set_adapters(model, ["task"], [expert], args.free_scale)
    with torch.no_grad():
        task_anchor, _ = action_logits(
            model, tokenizer, system, prompt, action, args.max_seq_len, False,
            assistant_prefix=prefix,
        )
        set_adapters(model, ["task", other], [expert], args.free_scale)
        other_logits, _ = action_logits(
            model, tokenizer, system, prompt, action, args.max_seq_len, False,
            assistant_prefix=prefix,
        )
        other_delta = (other_logits - task_anchor).detach()
    set_adapters(model, ["task", expert], [expert], args.free_scale)
    active_logits, _ = action_logits(
        model, tokenizer, system, prompt, action, args.max_seq_len, True,
        assistant_prefix=prefix,
    )
    cosine = F.cosine_similarity(
        (active_logits - task_anchor).reshape(1, -1), other_delta.reshape(1, -1)
    ).mean()
    separation = F.relu(cosine - args.sep_margin)
    (args.sep_coef * separation).backward()
    cosine_value = float(cosine.detach().cpu())
    separation_value = float(separation.detach().cpu())
    del task_anchor, other_logits, other_delta, active_logits, separation
    params = [p for p in model.parameters() if p.requires_grad]
    grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
    optimizers[expert].step()
    for opt in optimizers.values(): opt.zero_grad(set_to_none=True)
    total_routes = sum(route_counts.values())
    p1 = route_counts["free1"] / total_routes
    p2 = route_counts["free2"] / total_routes
    balance = sum(p * math.log(max(2 * p, 1e-12)) for p in (p1, p2) if p > 0)
    return {
        "train/block_return": gamma_return, "train/block_advantage": advantage,
        "train/active_expert": 1 if expert == "free1" else 2,
        "train/e2e_loss": sum(e2e_values) / len(e2e_values), "train/kl_loss": sum(kl_values) / len(kl_values),
        "train/separation_loss": separation_value, "train/delta_cosine": cosine_value,
        "train/balance_kl": balance, "train/free_grad_norm": float(grad),
        "train/free1_blocks": route_counts["free1"], "train/free2_blocks": route_counts["free2"],
    }


def evaluate_point(model, tokenizer, system, verbs, row, recent_actions, point_index, args):
    trace = row.get("f2p_trace") or {}
    prompt = trace.get("action_prompt") or ("My Current Observation:\n" + (row.get("observation") or {}).get("text", ""))
    paths = {
        "base": [],
        "task": ["task"],
        "free1_only": ["free1"],
        "free2_only": ["free2"],
        "task_free1": ["task", "free1"],
        "task_free2": ["task", "free2"],
    }
    outputs = {}
    probe_seed = args.seed + int(row.get("step", 0)) * 1000
    cpu_rng = torch.random.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    try:
        for name, adapters in paths.items():
            # Common random numbers make differences less sensitive to sampling
            # noise while retaining the real inference settings.
            torch.manual_seed(probe_seed)
            torch.cuda.manual_seed_all(probe_seed)
            outputs[name] = generate(model, tokenizer, system, prompt, adapters, args)
    finally:
        # Diagnostic probes must not perturb subsequent training stochasticity.
        torch.random.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
    actions = {k: v["action"] for k, v in outputs.items()}
    def valid(a): return int(bool(a) and any(a == verb or a.startswith(verb + " ") for verb in verbs))
    metrics = {"eval/point_index": point_index, "eval/unique_actions_six_paths": len(set(actions.values()))}
    for name, action in actions.items():
        metrics[f"eval/schema_valid_{name}"] = valid(action)
        metrics[f"eval/recent_repeat_{name}"] = int(action in recent_actions)
        metrics[f"eval/{name}_changes_historical"] = int(action != row.get("action"))
    metrics.update({
        "eval/task_free1_changes_task": int(actions["task_free1"] != actions["task"]),
        "eval/task_free2_changes_task": int(actions["task_free2"] != actions["task"]),
        "eval/combined_free_experts_differ": int(actions["task_free1"] != actions["task_free2"]),
    })
    payload = {
        "historical_step": row.get("step"), "observation": (row.get("observation") or {}).get("text", ""),
        "recent_historical_actions": list(recent_actions), "historical_action": row.get("action"),
        "historical_reward": row.get("reward"), "decisions": outputs,
    }
    return metrics, payload


def main():
    args = parse_args()
    if args.task_rank + args.free_rank != args.baseline_active_rank:
        raise ValueError(
            "Rank-matched invariant violated: task_rank + free_rank must equal "
            f"baseline_active_rank ({args.task_rank} + {args.free_rank} != "
            f"{args.baseline_active_rank})"
        )
    random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir).resolve(); output_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.source_log).resolve()
    rows = [json.loads(x) for x in source.read_text(encoding="utf-8").splitlines() if x.strip()]
    points = sorted({int(x) for x in args.points.split(",") if x.strip() and int(x) <= args.max_replay_step})
    config = vars(args).copy(); config.update({
        "algorithm": "rank_matched_sparse_trilora_v2",
        "active_rank": args.task_rank + args.free_rank,
        "stored_lora_rank": args.task_rank + 2 * args.free_rank,
        "six_path_turning_point_probe": True,
        "counterfactual_probe_used_for_training": False,
        "routing": "deterministic_balanced_top1_alternation", "causal_probe": True,
        "environment_inference_replayed": False, "old_checkpoint_loaded": False,
        "reward_scalar": "sum(non_death benchmark components)-death",
    })
    (output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    swan = SwanSidecar(output_dir, args.name, args.swanlab_group, config, not args.disable_swanlab)

    system, verbs = build_system_prompt()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    device_map = {"": 0} if args.device_map == "single" else "balanced"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device_map, trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    task_cfg = LoraConfig(
        r=args.task_rank, lora_alpha=2 * args.task_rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    free_cfg = LoraConfig(
        r=args.free_rank, lora_alpha=args.free_rank, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, task_cfg, adapter_name="task")
    model.add_adapter("free1", free_cfg); model.add_adapter("free2", free_cfg)
    set_adapters(model, ["task"], ["task"], args.free_scale)
    task_opt = AdamW([p for n, p in model.named_parameters() if ".task." in n], lr=args.task_lr)
    free_opts = {
        k: AdamW([p for n, p in model.named_parameters() if f".{k}." in n], lr=args.free_lr)
        for k in ("free1", "free2")
    }

    decision_path = output_dir / "turning_point_decisions.jsonl"
    train_path = output_dir / "training_metrics.jsonl"
    death_path = output_dir / "death_events.jsonl"
    free_data_paths = {
        name: output_dir / f"{name}_training_data.jsonl"
        for name in ("free1", "free2")
    }
    task_buffer, block, returns = [], [], []
    route_counts = Counter(); cumulative = Counter()
    recent_actions = deque(maxlen=25); point_index = 0; death_events = 0
    for row in rows:
        step = int(row.get("step", -1))
        if step > args.max_replay_step: break
        if step in points:
            metrics, payload = evaluate_point(model, tokenizer, system, verbs, row, recent_actions, point_index, args)
            with decision_path.open("a", encoding="utf-8") as f: f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            swan.log(step, metrics, payload); point_index += 1
            print("TURNING_POINT", step, {k: v["action"] for k, v in payload["decisions"].items()}, flush=True)

        # Stream every benchmark component and its cumulative value.  A death
        # additionally carries the exact before/assistant/after transcript.
        step_metrics = benchmark_metrics(row, cumulative)
        if float((row.get("reward") or {}).get("death", 0) or 0) > 0:
            death_events += 1
            event_payload = death_event_payload(row, recent_actions)
            with death_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event_payload, ensure_ascii=False) + "\n")
            swan.log(
                step, step_metrics, event_payload,
                text_key=f"events/death_step_{step}",
                caption=f"Death at historical step {step}: before / response / after",
            )
            print("DEATH_EVENT", step, event_payload.get("environment_time"), flush=True)
        else:
            swan.log(step, step_metrics)

        trace = row.get("f2p_trace") or {}
        prompt = trace.get("action_prompt") or ("My Current Observation:\n" + (row.get("observation") or {}).get("text", ""))
        task_buffer.append({"prompt": prompt, "action": str(row.get("action", "wait")), "w": float(trace.get("w_t", 0.0) or 0.0)})
        block.append(row); recent_actions.append(str(row.get("action", "")))
        train_metrics = {}
        if len(task_buffer) >= args.task_update_frequency:
            train_metrics.update(train_task(model, tokenizer, system, task_buffer, task_opt, args)); task_buffer = []
        if len(block) >= args.block_size:
            expert = "free1" if (sum(route_counts.values()) % 2 == 0) else "free2"
            train_metrics.update(train_free_block(
                model, tokenizer, system, block, expert, free_opts, returns,
                route_counts, free_data_paths[expert], sum(route_counts.values()), args,
            )); block = []
        if train_metrics:
            train_metrics["historical_step"] = step
            with train_path.open("a", encoding="utf-8") as f: f.write(json.dumps(train_metrics) + "\n")
            swan.log(step, train_metrics)
            print("TRAIN", step, train_metrics, flush=True)

    if task_buffer: train_task(model, tokenizer, system, task_buffer, task_opt, args)
    if block:
        expert = "free1" if (sum(route_counts.values()) % 2 == 0) else "free2"
        train_free_block(
            model, tokenizer, system, block, expert, free_opts, returns,
            route_counts, free_data_paths[expert], sum(route_counts.values()), args,
        )
    model.save_pretrained(output_dir / "adapters", selected_adapters=["task", "free1", "free2"])
    summary = {
        "turning_points_evaluated": point_index,
        "diagnostic_forwards_per_point": 6,
        "route_counts": dict(route_counts),
        "records_replayed": min(len(rows), args.max_replay_step + 1),
        "active_rank": args.task_rank + args.free_rank,
        "stored_lora_rank": args.task_rank + 2 * args.free_rank,
        "death_events": death_events,
        "benchmark_totals": {key: cumulative[key] for key in (*REWARD_KEYS, "score")},
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    swan.finish(); print("COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
