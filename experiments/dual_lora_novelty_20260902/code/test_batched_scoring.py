from __future__ import annotations

import os
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from novelty_patch import score_action_means_batched  # noqa: E402


class _TLM:
    pass


class _Agent:
    def _chat_prompt(self, outcome: str = "", base_prompt: str | None = None) -> str:
        return f"System prompt.\nState: {base_prompt}\nExact action:\n"


def _scalar_score(agent: _Agent, prompt: str, action: str) -> torch.Tensor:
    tokenizer = agent.tlm.tokenizer
    prompt_text = agent._chat_prompt("", base_prompt=prompt)
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt_text + action, add_special_tokens=False)["input_ids"]
    action_start = len(prompt_ids)
    keep = min(len(full_ids), agent.f2p_max_score_len)
    start = max(0, len(full_ids) - keep)
    ids = torch.tensor(full_ids[start:], dtype=torch.long).unsqueeze(0)
    first = min(max(0, action_start - start - 1), ids.shape[1] - 1)
    action_count = ids.shape[1] - 1 - first
    output = agent.tlm.model(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        logits_to_keep=action_count + 1,
    )
    logits = output.logits[0, -(action_count + 1) : -1].float()
    labels = ids[0, -action_count:]
    return (
        torch.log_softmax(logits, dim=-1)
        .gather(-1, labels.unsqueeze(-1))
        .mean()
    )


def test_batched_scores_match_individual_scores() -> None:
    model_path = os.environ.get("MODEL_PATH", "/home/sunward/models/Qwen3-4B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    torch.manual_seed(123)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            max_position_embeddings=256,
        )
    ).eval()
    agent = _Agent()
    agent.tlm = _TLM()
    agent.tlm.tokenizer = tokenizer
    agent.tlm.model = model
    agent.tlm.device = "cpu"
    agent.f2p_max_score_len = 128
    actions = ["wait", "attack goblin", "pick up the small bag"]
    batched = score_action_means_batched(
        agent, "test state", actions, requires_grad=False
    )
    individual = torch.stack(
        [_scalar_score(agent, "test state", action) for action in actions]
    )
    assert torch.allclose(batched, individual, rtol=2e-5, atol=2e-5)


if __name__ == "__main__":
    test_batched_scores_match_individual_scores()
    print("batched scoring tests passed")
