import time
import argparse
import os
import json
import shutil
import tempfile
import random
import copy
import torch
import torch.distributed as dist
from tools.logger import get_logger
from tools.swanlab_tracker import SwanLabTracker

from utils import dynamic_load_game_class, get_hardware_info, convert_json_to_jsonl


def _ensure_valid_actions(obs):
    if not isinstance(obs, dict):
        return
    for agent_id, agent_obs in obs.items():
        if not isinstance(agent_obs, dict):
            continue
        valid_actions = agent_obs.get("valid_actions")
        if valid_actions is None:
            agent_obs["valid_actions"] = {}
        elif not isinstance(valid_actions, dict):
            normalized = {"default": list(valid_actions) if isinstance(valid_actions, (list, tuple)) else []}
            agent_obs["valid_actions"] = {k: v for k, v in normalized.items() if v}
        else:
            cleaned = {}
            for key, value in list(valid_actions.items()):
                if value is None:
                    continue
                if not isinstance(value, list):
                    value = list(value) if isinstance(value, (tuple, set)) else [value]
                value = [item for item in value if item is not None and str(item).strip() != ""]
                if value:
                    cleaned[key] = value
            agent_obs["valid_actions"] = cleaned


def _parse_snapshot_steps(value):
    if value is None or not str(value).strip():
        return set()
    steps = set()
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step <= 0:
            raise ValueError("LoRA snapshot steps must be positive integers")
        steps.add(step)
    return steps


def _save_lora_snapshot(*, agent_dir, memory_dir, step, logger):
    source_dir = os.path.join(agent_dir, memory_dir, "lora")
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(
            f"Cannot save LoRA snapshot at step {step}: missing {source_dir}"
        )

    snapshot_root = os.path.join(agent_dir, "lora_checkpoints")
    snapshot_dir = os.path.join(snapshot_root, f"step_{step:04d}")
    os.makedirs(snapshot_root, exist_ok=True)
    if os.path.exists(snapshot_dir):
        logger.info(
            f"Keeping existing immutable LoRA snapshot at {snapshot_dir}"
        )
        return

    temporary_dir = tempfile.mkdtemp(
        prefix=f".step_{step:04d}_", dir=snapshot_root
    )
    try:
        shutil.copytree(source_dir, temporary_dir, dirs_exist_ok=True)
        with open(os.path.join(temporary_dir, "snapshot_metadata.json"), "w") as f:
            json.dump(
                {
                    "environment_step": step,
                    "source": source_dir,
                },
                f,
                indent=2,
            )
            f.write("\n")
        os.replace(temporary_dir, snapshot_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    logger.info(f"Saved immutable LoRA snapshot at {snapshot_dir}")

def build_agent_config(agent_type, common_cfg_kwargs):
    # Lazy-import agent config classes to avoid importing heavy/optional provider SDKs
    if agent_type in {"VanillaRAGAgent", "Mem0RAGAgent", "RaptorRAGAgent", "VoyagerAgent"}:
        from agents.rag.rag_agent_config import RAGAgentConfig
        return RAGAgentConfig(**common_cfg_kwargs)
    elif agent_type in {"VanillaParamAgent", "LoRASFTAgent", "EnvironmentPredictionTTTAgent", "FeedbackToPolicyTTTAgent", "LookaheadEnvironmentTTTAgent", "HindsightLongHorizonTTTAgent", "SparseTriLoRATTTAgent"}:
        from agents.parametric.param_agent_config import ParamAgentConfig
        return ParamAgentConfig(**common_cfg_kwargs)
    elif agent_type in {"LongContextAgent", "Mem1Agent", "ShortTermMemoryAgent"}:
        from agents.fixed_size.fixed_size_memory_agent_config import FixedSizeMemoryAgentConfig
        return FixedSizeMemoryAgentConfig(**common_cfg_kwargs)
    elif agent_type in {"MPlusAgent", "MemoryLLMAgent"}:
        from agents.latent.latent_agent_config import LatentAgentConfig
        return LatentAgentConfig(**common_cfg_kwargs)
    elif agent_type in {"NoMemoryAgent", "FakeLLMAgent"}:
        from agents.llm_agent_config import LLMAgentConfig
        return LLMAgentConfig(**common_cfg_kwargs)
    elif agent_type in {"ReplayAgent"}:
        return None
    return None

def instantiate_agent(agent_type, agent_info, AgentCls):
    """Create and return an agent instance of the given *agent_type*."""
    factory_map = {
        "HumanAgent":              ("agents.human_agent",                          "create_human_agent"),
        "VanillaRAGAgent":         ("agents.rag.vanilla_rag_agent",                "create_vanilla_rag_agent"),
        "Mem0RAGAgent":            ("agents.rag.mem0_rag_agent",                   "create_mem0_rag_agent"),
        "RaptorRAGAgent":          ("agents.rag.raptor_rag_agent",                 "create_raptor_rag_agent"),
        "VoyagerAgent":            ("agents.rag.voyager_agent",                    "create_voyager_agent"),
        "LoRASFTAgent":            ("agents.parametric.lora_sft_agent",            "create_lora_sft_agent"),
        "EnvironmentPredictionTTTAgent": ("agents.parametric.environment_prediction_ttt_agent", "create_environment_prediction_ttt_agent"),
        "FeedbackToPolicyTTTAgent": ("agents.parametric.feedback_to_policy_ttt_agent", "create_feedback_to_policy_ttt_agent"),
        "LookaheadEnvironmentTTTAgent": ("agents.parametric.lookahead_environment_ttt_agent", "create_lookahead_environment_ttt_agent"),
        "HindsightLongHorizonTTTAgent": ("agents.parametric.hindsight_long_horizon_ttt_agent", "create_hindsight_long_horizon_ttt_agent"),
        "SparseTriLoRATTTAgent": ("agents.parametric.sparse_trilora_ttt_agent", "create_sparse_trilora_ttt_agent"),
        "MPlusAgent":              ("agents.latent.mplus_agent",                   "create_mplus_agent"),
        "MemoryLLMAgent":          ("agents.latent.memoryllm_agent",               "create_memoryllm_agent"),
        "LongContextAgent":        ("agents.long_context_agent",                   "create_long_context_agent"),
        "ShortTermMemoryAgent":    ("agents.fixed_size.short_term_memory_agent",   "create_short_term_memory_agent"),
        "Mem1Agent":               ("agents.fixed_size.mem1_agent",               "create_mem1_agent"),
        "NoMemoryAgent":           ("agents.no_memory_agent",                      "create_no_memory_agent"),
        "RandomAgent":             ("agents.random_agent",                         "create_random_agent"),
        "WaitAgent":               ("agents.wait_agent",                           "create_wait_agent"),
        "FakeLLMAgent":            ("agents.fake_llm_agent",                       "create_fake_llm_agent"),
        "ReplayAgent":             ("agents.replay_agent",                         "create_replay_agent"),
    }
    if agent_type not in factory_map:
        raise ValueError(f"Unknown agent type: {agent_type}")
    module_path, factory_name = factory_map[agent_type]
    import importlib
    mod = importlib.import_module(module_path)
    factory_fn = getattr(mod, factory_name)
    DerivedCls = factory_fn(AgentCls)
    return DerivedCls(**agent_info)

def load_agents_config(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["agents"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # agents configuration file - for multi-agent settings, not required for single-agent runs
    parser.add_argument("--agents_config", type=str, default=None,
                        help="Path to a JSON file for agents configurations. "
                             "When provided, single-agent CLI flags (--agent, --agent_id, "
                             "--agent_name, --llm_name, --enable_*) are ignored.")

    # single-agent flags (used when --agents_config is not given)
    parser.add_argument("--agent", type=str, default="HumanAgent", help="The agent to evaluate")
    parser.add_argument("--agent_id", type=str, default="agent_adam_davis", help="Unique id for the agent")
    parser.add_argument("--agent_name", type=str, default="adam_davis", help="Human readable agent name")
    parser.add_argument("--llm_provider", type=str, default=None, choices=["openai", "huggingface", "vllm", "azure", "azure_openai", "claude", "gemini"], help="Which LLM provider to use")
    parser.add_argument("--llm_name", type=str, default=None, help="Name of the LLM to use")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override llm_name for all configured agents; may also be set with AGENTODYSSEY_MODEL_PATH.")
    parser.add_argument("--enable_short_term_memory", action="store_true")
    parser.add_argument("--short_term_memory_size", type=int, default=5)
    parser.add_argument("--enable_reflection", action="store_true")
    parser.add_argument("--enable_summarization", action="store_true")

    # env flags
    parser.add_argument("--game_name", type=str, default="base", help="Which game to run: base or a folder under games/generated/")
    parser.add_argument("--world_definition_path", type=str, default=None, help="Path to the world definition JSON file, auto routing if not specified")
    parser.add_argument("--env_config_path", type=str, default=None, help="Path to the initial env config JSON/JSONL file, auto routing if not specified")
    parser.add_argument("--output_dir", type=str, default="output", help="Path to the general output directory")
    parser.add_argument("--run_dir", type=str, default=None, help="Path storing the current run's data; if None, a new directory will be created under output_dir")
    parser.add_argument("--extra_dir", type=str, default=None, help="Adding an extra directory level under run_dir for multiple runs over same configuration")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite the run directory if it exists")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--max_steps", type=int, default=300, help="Maximum number of steps to run in the environment")
    parser.add_argument("--enforce_same_hardware", action="store_true", help="Whether to enforce the same hardware for resumed runs")
    parser.add_argument("--enable_obs_valid_actions", action="store_true", help="Whether to include valid actions in the observation")  # required for RandomAgent
    parser.add_argument("--cumulative_config_save", action="store_true", help="Save cumulative env config each step")
    parser.add_argument("--debug", action="store_true", help="(Deprecated) Alias for enabling both --cumulative_agent_log and --cumulative_config_save")
    parser.add_argument("--resume_from_step", type=int, default=None, help="If specified, resume from the given step number")
    parser.add_argument("--save_dep_graph_steps", type=int, default=None, help="Number of steps before a new dependency graph will be saved; if None, dependency tracking will be disabled")

    parser.add_argument("--memory_dir", type=str, default="memory", help="Directory to save agent memory checkpoints under run_dir")
    parser.add_argument("--agent_memory_save_frequency", type=int, default=1,
                        help="Save agent memory every N environment steps. If None, disabled.")
    parser.add_argument(
        "--agent_lora_snapshot_steps",
        type=str,
        default="",
        help=(
            "Comma-separated environment steps at which the just-saved LoRA "
            "directory is copied to an immutable lora_checkpoints/step_NNNN snapshot."
        ),
    )
    parser.add_argument("--disable_swanlab", action="store_true",
                        help="Disable the default online SwanLab tracking.")
    parser.add_argument("--swanlab_project", type=str, default="agentic-TTT")
    parser.add_argument("--swanlab_workspace", type=str, default="ZitongWang")
    parser.add_argument("--swanlab_experiment_name", type=str, default=None)
    parser.add_argument("--swanlab_group", type=str, default=None)
    args = parser.parse_args()
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed SparseTriLoRA requires CUDA")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    lora_snapshot_steps = _parse_snapshot_steps(args.agent_lora_snapshot_steps)
    if any(step > args.max_steps for step in lora_snapshot_steps):
        raise ValueError(
            "LoRA snapshot steps cannot exceed --max_steps: "
            f"{sorted(lora_snapshot_steps)} vs {args.max_steps}"
        )

    if args.debug:
        args.cumulative_config_save = True

    if args.game_name in (None, "", "base"):
        world_definition_path = f"assets/world_definitions/base/default.json"
        env_config_path = f"assets/env_configs/base/initial.json"
    else:
        world_definition_path = f"assets/world_definitions/generated/{args.game_name}/default.json"
        env_config_path = f"assets/env_configs/generated/{args.game_name}/initial.json"

    if args.world_definition_path is not None:
        world_definition_path = args.world_definition_path
    if args.env_config_path is not None:
        env_config_path = args.env_config_path

    logger = get_logger("EvalLogger")

    if args.agents_config is not None:
        agent_specs = load_agents_config(args.agents_config)
    else:
        agent_specs = [{
            "agent_type": args.agent,
            "agent_id": args.agent_id,
            "agent_name": args.agent_name,
            "llm_name": args.llm_name,
            "llm_provider": args.llm_provider,
            "enable_short_term_memory": args.enable_short_term_memory,
            "short_term_memory_size": args.short_term_memory_size,
            "enable_reflection": args.enable_reflection,
            "enable_summarization": args.enable_summarization,
        }]

    model_override = args.model_path or os.environ.get("AGENTODYSSEY_MODEL_PATH")
    if model_override:
        agent_specs = [dict(spec, llm_name=model_override) for spec in agent_specs]

    run_dir = args.run_dir if args.run_dir is not None else os.path.join(args.output_dir, "game_" + args.game_name)
    if args.agents_config is None:
        if args.llm_name is not None:
            run_dir = os.path.join(run_dir, args.llm_name.replace("/", "_"))
        run_dir = os.path.join(run_dir, args.agent)
        if args.enable_short_term_memory:
            run_dir = os.path.join(run_dir, "with_short_term_memory")
        elif args.enable_reflection:
            run_dir = os.path.join(run_dir, "with_reflection")
        elif args.enable_summarization:
            run_dir = os.path.join(run_dir, "with_summarization")
        else:
            run_dir = os.path.join(run_dir, "no_extras")
    if args.extra_dir is not None:
        run_dir = os.path.join(run_dir, args.extra_dir)

    if args.resume_from_step and (not os.path.exists(run_dir) or not args.cumulative_config_save):
        raise ValueError("To resume from a specific step, the run_dir must exist and --cumulative_config_save must be enabled.")
    if args.overwrite and os.path.exists(run_dir):
        logger.warning(f"Overwriting the run directory: {run_dir}")
        shutil.rmtree(run_dir)
    os.makedirs(run_dir, exist_ok=True)

    AgentCls = dynamic_load_game_class(args.game_name, "agent", "Agent")
    agents = []
    agent_dirs = {}
    agent_log_paths = {}

    for spec in agent_specs:
        a_type = spec["agent_type"]
        a_id   = spec["agent_id"]
        a_name = spec["agent_name"]
        a_llm  = spec.get("llm_name")
        a_prov = spec.get("llm_provider")
        a_stm  = spec.get("enable_short_term_memory", False)
        a_stms = spec.get("short_term_memory_size", 5)
        a_ref  = spec.get("enable_reflection", False)
        a_sum  = spec.get("enable_summarization", False)

        agent_dir = os.path.join(run_dir, a_id)
        os.makedirs(os.path.join(agent_dir, args.memory_dir), exist_ok=True)
        agent_dirs[a_id] = agent_dir
        agent_log_paths[a_id] = os.path.join(agent_dir, "agent_log.jsonl")

        common_cfg_kwargs = dict(
            llm_name=a_llm,
            llm_provider=a_prov,
            enable_reflection=a_ref,
            enable_summarization=a_sum,
            enable_short_term_memory=a_stm,
            short_term_memory_size=a_stms,
            full_mem_path=os.path.join(agent_dir, args.memory_dir),
        )

        # Optional parametric-agent knobs.  Keeping these in the agent spec
        # makes online SFT/LoRA runs auditable and reproducible without
        # changing the defaults of the other agent families.
        if a_type in {"VanillaParamAgent", "LoRASFTAgent", "EnvironmentPredictionTTTAgent", "FeedbackToPolicyTTTAgent", "LookaheadEnvironmentTTTAgent", "HindsightLongHorizonTTTAgent", "SparseTriLoRATTTAgent"}:
            for key in (
                "max_seq_len",
                "max_new_tokens",
                "temperature",
                "top_p",
                "lr",
                "epochs",
                "batch_size",
                "grad_accum",
                "fp16",
                "f2p_beta",
                "f2p_update_frequency",
                "hindsight_horizon",
                "prompt_memory_max_chars",
                "policy_update_frequency",
                "policy_lr",
                "policy_epochs",
                "task_rank",
                "free_rank",
                "free_scale",
                "free_block_horizon",
                "free_gamma",
                "free_lr",
                "free_kl_coef",
                "free_sep_coef",
                "free_sep_margin",
                "trilora_diagnostic_points",
            ):
                if key in spec:
                    common_cfg_kwargs[key] = spec[key]

        cfg = build_agent_config(a_type, common_cfg_kwargs)
        agent_info = {"id": a_id, "name": a_name}
        if cfg is not None:
            agent_info["cfg"] = cfg
            logger.info(f"Agent {a_id} ({a_type}) config: {cfg}")

        agent = instantiate_agent(a_type, agent_info, AgentCls)
        if a_type == "FeedbackToPolicyTTTAgent":
            agent.f2p_log_path = os.path.join(agent_dir, "f2p_intermediates.jsonl")
        if a_type == "LookaheadEnvironmentTTTAgent":
            agent.lookahead_log_path = os.path.join(agent_dir, "lookahead_intermediates.jsonl")
        if a_type == "HindsightLongHorizonTTTAgent":
            agent.hindsight_log_path = os.path.join(agent_dir, "hindsight_intermediates.jsonl")
        if a_type == "SparseTriLoRATTTAgent":
            agent.f2p_log_path = os.path.join(agent_dir, "f2p_intermediates.jsonl")
            agent.trilora_log_path = os.path.join(agent_dir, "trilora_intermediates.jsonl")
            agent.diagnostic_log_path = os.path.join(agent_dir, "trilora_diagnostics.jsonl")
            agent.free_data_paths = {
                "free1": os.path.join(agent_dir, "free1_training_data.jsonl"),
                "free2": os.path.join(agent_dir, "free2_training_data.jsonl"),
            }
        agents.append(agent)

    # Rank 0 alone owns the stateful environment. Other torchrun ranks are
    # synchronous model workers for diagnostics and data-parallel TTT updates.
    if distributed and dist.get_rank() != 0:
        if len(agents) != 1 or not hasattr(agents[0], "distributed_worker_loop"):
            raise RuntimeError("torchrun worker mode requires one distributed-capable agent")
        agents[0].distributed_worker_loop()
        dist.destroy_process_group()
        raise SystemExit(0)

    logger.info(f"Created {len(agents)} agent(s): {[a.id for a in agents]}")
    for a_id, a_dir in agent_dirs.items():
        logger.info(f"{a_id} memory dir: {os.path.join(a_dir, args.memory_dir)}")
    logger.info(f"Agent memory save frequency: {args.agent_memory_save_frequency}")
    logger.info(f"Immutable LoRA snapshot steps: {sorted(lora_snapshot_steps)}")

    config_path = os.path.join(run_dir, "config.json")
    if args.cumulative_config_save:
        config_path = os.path.join(run_dir, "config.jsonl")

    from_config = False
    if not os.path.exists(config_path):
        seed_config_path = env_config_path
        logger.info(f"Initiating new game from the world config: {world_definition_path}")
        logger.info(f"Initiating new game from the environment config: {seed_config_path}")
        if args.cumulative_config_save:
            convert_json_to_jsonl(seed_config_path, config_path)
        else:
            shutil.copy(seed_config_path, config_path)
    else:
        logger.info(f"Continuing the game from config: {config_path}")
        from_config = True

    args.seed = args.seed if args.seed is not None else random.randint(0, 10000)

    EnvCls = dynamic_load_game_class(args.game_name, "env", "AgentOdysseyEnv")
    env = EnvCls(
        seed=args.seed,
        agents=agents,
        world_definition_path=world_definition_path,
        run_dir=run_dir,
        config_path=config_path,
        enable_obs_valid_actions=args.enable_obs_valid_actions,
        from_step=args.resume_from_step,
        save_dep_graph_steps=args.save_dep_graph_steps,
    )
    
    this_hardware = get_hardware_info()
    if not args.overwrite and args.enforce_same_hardware and "hardware" in env.config:
        saved_hardware = env.config["hardware"]
        for key in this_hardware:
            if this_hardware[key] != saved_hardware.get(key, None):
                raise ValueError(f"The current hardware {key}: {this_hardware[key]} does not match the saved hardware {key}: {saved_hardware[key]}. Cannot resume the run.")
        logger.info("The current hardware matches the saved hardware. Resuming the run...")
    else:
        env.config["hardware"] = this_hardware

    if not args.overwrite:
        for arg_name, arg_val in [("game_name", args.game_name), ("world_definition_path", world_definition_path), ("seed", args.seed)]:
            if arg_name in env.config and env.config[arg_name] != arg_val:
                raise ValueError(f"The current {arg_name}: {arg_val} does not match the saved {arg_name}: {env.config[arg_name]}. Cannot resume the run.")
    env.config["game_name"] = args.game_name
    env.config["world_definition_path"] = world_definition_path
    env.config["seed"] = args.seed
    env.update_config(update_env_config=False, hardware=env.config["hardware"], update_file=False, cumulative=args.cumulative_config_save)

    obs = env.reset(from_config=from_config)
    env.update_config(update_env_config=True, update_file=False, cumulative=args.cumulative_config_save)
    if args.enable_obs_valid_actions:
        _ensure_valid_actions(obs)

    # load per-agent memory from each agent's own directory
    for agent in env.agents:
        if hasattr(agent, "load_memory"):
            a_dir = agent_dirs[agent.id]
            full_memory_paths = [os.path.join(a_dir, args.memory_dir, mp) for mp in agent.memory_paths]
            if all(os.path.exists(p) for p in full_memory_paths):
                agent.load_memory(full_memory_dir=os.path.join(a_dir, args.memory_dir))
                logger.info(f"Loaded memory for agent {agent.id} from {full_memory_paths}.")
            else:
                logger.info(f"No previous memory found for agent {agent.id}. Starting fresh.")

    swan_tracker = SwanLabTracker(
        run_dir=run_dir,
        args=args,
        agent_specs=agent_specs,
        hardware=this_hardware,
        logger=logger,
    )

    while True:
        action_strs = {}
        agents_log = {}
        # collect per-agent outputs from act()
        for agent in env.agents:
            start_time = time.perf_counter()
            action_strs[agent.id], num_input_tokens, num_output_tokens, response = agent.act(obs[agent.id])
            end_time = time.perf_counter()
            decision_time = end_time - start_time
            agents_log[agent.id] = {
                "observation": obs[agent.id],
                "action": action_strs[agent.id],
                "num_input_tokens": num_input_tokens,
                "num_output_tokens": num_output_tokens,
                "decision_time": decision_time,
                "response": response,
            }

        try:
            obs, reward, done, info = env.step(action_strs)
            if args.enable_obs_valid_actions:
                _ensure_valid_actions(obs)
        except Exception as e:
            logger.error(f"Error during env.step: {e}; Agent actions: {action_strs}")
            raise

        # All parametric TTT agents consume the real transition after the
        # environment advances.  Keeping this hook in the shared evaluator
        # prevents silent ordinary-evaluation runs from TTT configurations.
        for agent in env.agents:
            if hasattr(agent, "observe_transition"):
                agent.observe_transition(
                    previous_obs=agents_log[agent.id]["observation"],
                    action=action_strs[agent.id],
                    next_obs=obs[agent.id],
                    reward=reward.get(agent.id) if isinstance(reward, dict) else None,
                    info=info,
                )
                if hasattr(agent, "get_f2p_trace"):
                    agents_log[agent.id]["f2p_trace"] = agent.get_f2p_trace()
                if hasattr(agent, "get_lookahead_trace"):
                    agents_log[agent.id]["lookahead_trace"] = agent.get_lookahead_trace()
                if hasattr(agent, "get_hindsight_trace"):
                    agents_log[agent.id]["hindsight_trace"] = agent.get_hindsight_trace()
                if hasattr(agent, "get_trilora_trace"):
                    agents_log[agent.id]["trilora_trace"] = agent.get_trilora_trace()

        episode_finished = bool(done or env.steps >= args.max_steps)
        if episode_finished:
            # Flush tail windows before writing the final step, so the final
            # agent log observes the same completed lifecycle as the reports.
            for agent in env.agents:
                if hasattr(agent, "finish_episode"):
                    agent.finish_episode()
                for trace_name, getter_name in (
                    ("f2p_trace", "get_f2p_trace"),
                    ("lookahead_trace", "get_lookahead_trace"),
                    ("hindsight_trace", "get_hindsight_trace"),
                    ("trilora_trace", "get_trilora_trace"),
                ):
                    if hasattr(agent, getter_name):
                        agents_log[agent.id][trace_name] = getattr(agent, getter_name)()

        # update scores in env
        env.update_scores(reward)

        invalids = info.get("step_invalid_action", {}) if isinstance(info, dict) else {}
        for agent in env.agents:
            log_path = agent_log_paths[agent.id]
            combined = {
                "step": env.steps - 1,
                "action": action_strs.get(agent.id),
                "decision_time": agents_log[agent.id]["decision_time"],
                "num_input_tokens": agents_log[agent.id]["num_input_tokens"],
                "num_output_tokens": agents_log[agent.id]["num_output_tokens"],
                "invalid_action": bool(invalids.get(agent.id, False)),
                "reward": copy.deepcopy(reward[agent.id].__dict__),
                "observation": agents_log[agent.id].get("observation"),
                "response": agents_log[agent.id].get("response"),
            }
            for trace_name in ("f2p_trace", "lookahead_trace", "hindsight_trace", "trilora_trace"):
                if trace_name in agents_log[agent.id]:
                    combined[trace_name] = agents_log[agent.id][trace_name]
            with open(log_path, "a") as f:
                f.write(json.dumps(combined) + "\n")
            swan_tracker.log_step(
                agent_id=agent.id,
                record=combined,
                next_observation=obs.get(agent.id),
            )

        env.update_config(cumulative=args.cumulative_config_save)

        # Save only after the real transition, F2P update, log, and environment
        # state are complete.  A resumed run therefore never combines the
        # previous environment state with a newer F2P buffer/adapter.
        checkpoint_due = (
            args.agent_memory_save_frequency is not None
            and args.agent_memory_save_frequency > 0
            and (
                episode_finished
                or env.steps % args.agent_memory_save_frequency == 0
            )
        )
        if checkpoint_due:
            for agent in env.agents:
                a_dir = agent_dirs[agent.id]
                if hasattr(agent, "save_memory"):
                    agent.save_memory(
                        full_memory_dir=os.path.join(a_dir, args.memory_dir)
                    )
                    logger.info(f"Saving memory for {agent.id} at step {env.steps}")
                    if env.steps in lora_snapshot_steps:
                        _save_lora_snapshot(
                            agent_dir=a_dir,
                            memory_dir=args.memory_dir,
                            step=env.steps,
                            logger=logger,
                        )
                else:
                    logger.warning(
                        f"Agent {agent.id} does not have save_memory method. "
                        "Skipping memory save."
                    )

        if episode_finished:
            swan_tracker.finish()
            for agent in env.agents:
                if hasattr(agent, "stop_distributed_workers"):
                    agent.stop_distributed_workers()
            if distributed and dist.is_initialized():
                dist.destroy_process_group()
            logger.info(f"Episode finished after {env.steps} steps")
            break
