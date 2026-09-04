"""Optional, centralized SwanLab tracking for AgentOdyssey evaluations."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
from collections import Counter, deque
from datetime import datetime, timezone


REWARD_KEYS = (
    "quest",
    "exploration",
    "craft",
    "kill",
    "unique_kill",
    "side_quest",
    "trade",
    "death",
)


def _numeric_trace(prefix, value, output, depth=0):
    """Flatten useful numeric trace leaves without uploading large text blobs."""
    if depth > 5:
        return
    if isinstance(value, bool):
        output[prefix] = int(value)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        output[prefix] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            _numeric_trace(f"{prefix}/{key}", child, output, depth + 1)


def _compact_interaction(record, next_observation=None):
    if record is None:
        return None
    observation = record.get("observation") or {}
    response = record.get("response")
    if isinstance(response, dict):
        response = response.get("response", response)
    return {
        "step": record.get("step"),
        "observation_before": observation.get("text", observation),
        "model_response": response,
        "action": record.get("action"),
        "invalid_action": record.get("invalid_action"),
        "reward": record.get("reward"),
        "observation_after": (
            next_observation.get("text", next_observation)
            if isinstance(next_observation, dict)
            else next_observation
        ),
    }


class SwanLabTracker:
    def __init__(self, *, run_dir, args, agent_specs, hardware, logger):
        self.run_dir = os.path.abspath(run_dir)
        self.logger = logger
        self.enabled = False
        self._finished = False
        self._sidecar = None
        self._sidecar_log_handle = None
        self._history = {}
        self._pending_deaths = {}
        self._cumulative_rewards = {}
        self._invalid_counts = Counter()
        self._action_counts = {}
        self._recent_actions = {}
        self._death_event_path = os.path.join(self.run_dir, "death_events.jsonl")

        if getattr(args, "disable_swanlab", False):
            logger.info("SwanLab tracking disabled by --disable_swanlab")
            return
        sidecar_python = os.environ.get(
            "SWANLAB_SIDECAR_PYTHON", "/root/autodl-tmp/swanlab-venv/bin/python"
        )
        sidecar_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "swanlab_sidecar.py"
        )
        if not os.path.isfile(sidecar_python) or not os.path.isfile(sidecar_script):
            logger.warning(
                "SwanLab sidecar unavailable; continuing without cloud tracking: "
                f"python={sidecar_python}, script={sidecar_script}"
            )
            return
        experiment_name = getattr(args, "swanlab_experiment_name", None)
        if not experiment_name:
            agent_ids = "-".join(str(spec.get("agent_id", "agent")) for spec in agent_specs)
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            experiment_name = f"{args.game_name}-{agent_ids}-{stamp}"

        config = {
            "game_name": args.game_name,
            "seed": args.seed,
            "max_steps": args.max_steps,
            "run_dir": self.run_dir,
            "model_path": args.model_path,
            "agents_config": args.agents_config,
            "agent_specs": agent_specs,
            "hardware": hardware,
        }
        try:
            config["git_commit"] = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            pass

        init_kwargs = {
            "project": getattr(args, "swanlab_project", "agentic-TTT"),
            "workspace": getattr(args, "swanlab_workspace", "ZitongWang"),
            "experiment_name": experiment_name,
            "group": getattr(args, "swanlab_group", None) or args.game_name,
            "tags": [args.game_name, "agentodyssey", "ttt"],
            "config": config,
            "logdir": os.path.join(self.run_dir, "swanlog"),
            "mode": "online",
        }
        try:
            sidecar_log_path = os.path.join(self.run_dir, "swanlab-sidecar.log")
            self._sidecar_log_handle = open(
                sidecar_log_path, "a", encoding="utf-8", buffering=1
            )
            sidecar_env = os.environ.copy()
            # /etc/network_turbo is useful for GitHub/Hugging Face, but its
            # proxy is slower and less reliable for the SwanLab API.
            for key in (
                "http_proxy", "https_proxy", "all_proxy",
                "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            ):
                sidecar_env.pop(key, None)
            self._sidecar = subprocess.Popen(
                [sidecar_python, sidecar_script],
                stdin=subprocess.PIPE,
                stdout=self._sidecar_log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=sidecar_env,
            )
            self._send(
                {
                    "type": "init",
                    "kwargs": init_kwargs,
                    "run_id_path": os.path.join(self.run_dir, ".swanlab_run_id"),
                }
            )
            self.enabled = True
            atexit.register(self.finish)
            logger.info(
                "SwanLab sidecar tracking enabled: "
                f"ZitongWang/agentic-TTT ({experiment_name}); log={sidecar_log_path}"
            )
        except Exception as exc:
            logger.warning(f"SwanLab initialization failed; local evaluation continues: {exc}")

    def _send(self, payload):
        if self._sidecar is None or self._sidecar.stdin is None:
            return
        self._sidecar.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._sidecar.stdin.flush()

    def _write_death_event(self, payload):
        with open(self._death_event_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _log_death_context(self, agent_id, payload, step):
        self._write_death_event(payload)
        if self.enabled:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            self._send(
                {
                    "type": "log",
                    "step": step,
                    "metrics": {"death/event": 1},
                    "texts": {
                        "death/context": {
                            "data": text,
                            "caption": f"Death at environment step {payload['death_step']}",
                        }
                    },
                }
            )

    def log_step(self, *, agent_id, record, next_observation):
        step = int(record["step"])

        pending = self._pending_deaths.pop(agent_id, None)
        if pending is not None:
            pending["next_interaction"] = _compact_interaction(record, next_observation)
            self._log_death_context(agent_id, pending, step)

        reward = record.get("reward") or {}
        cumulative = self._cumulative_rewards.setdefault(
            agent_id, {key: 0.0 for key in REWARD_KEYS}
        )
        for key in REWARD_KEYS:
            cumulative[key] += float(reward.get(key, 0) or 0)

        action = str(record.get("action") or "")
        counts = self._action_counts.setdefault(agent_id, Counter())
        recent = self._recent_actions.setdefault(agent_id, deque(maxlen=25))
        repeated_exact = int(counts[action] > 0) if action else 0
        repeated_local = int(action in recent) if action else 0
        if action:
            counts[action] += 1
            recent.append(action)

        self._invalid_counts[agent_id] += int(bool(record.get("invalid_action")))
        positive = sum(cumulative[key] for key in REWARD_KEYS if key != "death")
        # Match the historical importer exactly so all Remnant runs overlay in
        # one SwanLab chart namespace. The agent id stays in run config/events.
        metrics = {
            "system/decision_seconds": record.get("decision_time", 0),
            "tokens/input": record.get("num_input_tokens", 0),
            "tokens/output": record.get("num_output_tokens", 0),
            "behavior/invalid": int(bool(record.get("invalid_action"))),
            "behavior/invalid_cumulative": self._invalid_counts[agent_id],
            "behavior/unique_actions": len(counts),
            "behavior/exact_repeat": repeated_exact,
            "behavior/local25_repeat": repeated_local,
            "benchmark/positive_reward": positive,
            "benchmark/net_reward_proxy": positive - cumulative["death"],
            "environment/original_step": step,
        }
        for key in REWARD_KEYS:
            metrics[f"reward_step/{key}"] = float(reward.get(key, 0) or 0)
            metrics[f"benchmark/{key}"] = cumulative[key]
        for trace_name in ("f2p_trace", "lookahead_trace", "hindsight_trace", "trilora_trace"):
            if trace_name in record:
                _numeric_trace(f"trace/{trace_name}", record[trace_name], metrics)

        if self.enabled:
            self._send({"type": "log", "step": step, "metrics": metrics})

        if float(reward.get("death", 0) or 0) > 0:
            payload = {
                "event": "agent_death",
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "agent_id": agent_id,
                "death_step": step,
                "previous_interaction": self._history.get(agent_id),
                "death_interaction": _compact_interaction(record, next_observation),
                "next_interaction": None,
            }
            # Upload immediately, then upload an enriched version after the next step.
            self._log_death_context(agent_id, payload, step)
            self._pending_deaths[agent_id] = payload

        self._history[agent_id] = _compact_interaction(record, next_observation)

    def finish(self):
        if self._finished:
            return
        self._finished = True
        for agent_id, payload in list(self._pending_deaths.items()):
            self._log_death_context(agent_id, payload, int(payload["death_step"]))
        self._pending_deaths.clear()
        if self.enabled and self._sidecar is not None:
            try:
                self._send({"type": "finish"})
                if self._sidecar.stdin is not None:
                    self._sidecar.stdin.close()
                self._sidecar.wait(timeout=5)
            except Exception as exc:
                self.logger.warning(f"SwanLab finish failed: {exc}")
            finally:
                if self._sidecar_log_handle is not None:
                    self._sidecar_log_handle.close()
