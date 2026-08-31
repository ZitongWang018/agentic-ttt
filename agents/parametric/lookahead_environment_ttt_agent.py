from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Any, Dict, Type

from agents.parametric.environment_prediction_ttt_agent import EnvironmentPredictionTTTAgent


class LookaheadEnvironmentTTTAgent(EnvironmentPredictionTTTAgent):
    """Minimal lookahead extension of EnvironmentPredictionTTT.

    The only policy change is:
        history -> initial action -> predicted outcome -> revised action.
    The revised action is executed.  A final outcome prediction is recorded for
    that executed action, while training remains the original observed-outcome
    loss from EnvironmentPredictionTTT.
    """

    def __init__(self, id: str, name: str, cfg=None, train_cfg=None):
        super().__init__(id=id, name=name, cfg=cfg, train_cfg=train_cfg)
        self.lookahead_log_path: str | None = None
        self.lookahead_step = 0
        self.lookahead_last_trace: Dict[str, Any] = {}
        self._revision_prompt = ""
        self._final_prediction_prompt = ""

    @staticmethod
    def _parse(parser, response: str) -> Dict[str, Any]:
        parsed = parser(response)
        if not isinstance(parsed, dict):
            return {"reasoning": "parse failure", "action": "wait", "predicted_outcome": ""}
        action = parsed.get("action", "wait")
        if not isinstance(action, str) or not action.strip():
            parsed["action"] = "wait"
        else:
            parsed["action"] = action.strip()
        parsed["predicted_outcome"] = str(parsed.get("predicted_outcome", ""))
        return parsed

    def _act(self, obs: Dict[str, Any]):
        obs_text = obs.get("text", "")
        memory_text = "\n\n".join(self.short_term_memory)
        base_prompt = (
            "My Current Observation:\n" + obs_text
            + ("\n\nVerified recent transitions:\n" + memory_text if memory_text else "")
            + "\n\nPropose one valid action and predict its immediate environment change."
        )
        first = self.tlm.generate(user_prompt=base_prompt, system_prompt=self._prediction_system)
        first_parsed = self._parse(self.cfg.json_parser, first["response"])
        initial_action = first_parsed["action"]
        initial_prediction = first_parsed["predicted_outcome"]

        self._revision_prompt = (
            base_prompt
            + "\n\nYour initial proposed action was:\n" + initial_action
            + "\nYour predicted immediate outcome was:\n" + initial_prediction
            + "\n\nNow reconsider the action using that predicted outcome. Choose the single valid action that should actually be executed."
        )
        revised = self.tlm.generate(user_prompt=self._revision_prompt, system_prompt=self._prediction_system)
        revised_parsed = self._parse(self.cfg.json_parser, revised["response"])
        final_action = revised_parsed["action"]

        self._final_prediction_prompt = (
            self._revision_prompt
            + "\n\nThe final action to execute is fixed as:\n" + final_action
            + "\n\nPredict only the immediate environment change caused by this fixed action."
        )
        final_prediction_output = self.tlm.generate(
            user_prompt=self._final_prediction_prompt,
            system_prompt=self._prediction_system,
        )
        final_prediction_parsed = self._parse(self.cfg.json_parser, final_prediction_output["response"])
        executed_prediction = final_prediction_parsed["predicted_outcome"]

        self._last_prompt = self._final_prediction_prompt
        self._last_prediction = executed_prediction
        self._last_action = final_action
        self._last_lookahead_bundle = {
            "initial_action": initial_action,
            "initial_predicted_outcome": initial_prediction,
            "revised_action": final_action,
            "revised_predicted_outcome": revised_parsed["predicted_outcome"],
            "executed_action_prediction": executed_prediction,
            "initial_response": first["response"],
            "revision_response": revised["response"],
            "final_prediction_response": final_prediction_output["response"],
        }
        response = json.dumps(self._last_lookahead_bundle, ensure_ascii=False)
        return (
            final_action,
            first["num_input_tokens"] + revised["num_input_tokens"] + final_prediction_output["num_input_tokens"],
            first["num_output_tokens"] + revised["num_output_tokens"] + final_prediction_output["num_output_tokens"],
            response,
        )

    def observe_transition(self, previous_obs, action: str, next_obs, reward=None, info=None):
        before = self.steps_trained_total
        super().observe_transition(previous_obs, action, next_obs, reward=reward, info=info)
        self.lookahead_step += 1
        trace = {
            "step": self.lookahead_step - 1,
            **getattr(self, "_last_lookahead_bundle", {}),
            "executed_action": action,
            "previous_observation": previous_obs.get("text", "") if isinstance(previous_obs, dict) else str(previous_obs),
            "real_observation": next_obs.get("text", "") if isinstance(next_obs, dict) else str(next_obs),
            "training_steps_total_before": before,
            "training_steps_total_after": self.steps_trained_total,
            "training_triggered": self.steps_trained_total > before,
            "training_objective": "-log p_theta(real_next_observation | history, executed_action)",
        }
        self.lookahead_last_trace = trace
        if self.lookahead_log_path:
            os.makedirs(os.path.dirname(self.lookahead_log_path), exist_ok=True)
            with open(self.lookahead_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(trace, ensure_ascii=False) + "\n")

    def get_lookahead_trace(self):
        return self.lookahead_last_trace


@lru_cache(maxsize=None)
def create_lookahead_environment_ttt_agent(Agent: Type):
    class_name = f"LookaheadEnvironmentTTTAgent__{Agent.__module__}.{Agent.__name__}"
    return type(
        class_name,
        (LookaheadEnvironmentTTTAgent, Agent),
        {"__module__": Agent.__module__, "__agent__": Agent},
    )
