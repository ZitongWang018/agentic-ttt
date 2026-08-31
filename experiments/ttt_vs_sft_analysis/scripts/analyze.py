import json
import math
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
FIG = ROOT / "figures"
OUT = ROOT / "outputs"
FIG.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)
COLORS = {"Environment-prediction TTT": "#1f77b4", "SFT + Short-Term Memory": "#d62728", "Feedback-to-Policy TTT (Remnant)": "#2ca02c", "Feedback-to-Policy TTT (Metropolis)": "#ff7f0e", "Lookahead Environment-TTT (Remnant)": "#9467bd", "Lookahead Environment-TTT (Metropolis)": "#8c564b", "Hindsight Long-Horizon TTT (Metropolis)": "#e377c2"}
LABELS = {"Environment-prediction TTT": "Environment-TTT", "SFT + Short-Term Memory": "SFT+Memory", "Feedback-to-Policy TTT (Remnant)": "F2P-TTT (Remnant)", "Feedback-to-Policy TTT (Metropolis)": "F2P-TTT (Metropolis)", "Lookahead Environment-TTT (Remnant)": "Lookahead Env-TTT (Remnant)", "Lookahead Environment-TTT (Metropolis)": "Lookahead Env-TTT (Metropolis)", "Hindsight Long-Horizon TTT (Metropolis)": "Hindsight Long-Horizon TTT", "Environment-prediction TTT (Remnant)": "Environment-TTT (Remnant)", "SFT + Short-Term Memory (Remnant)": "SFT+Memory (Remnant)", "Environment-prediction TTT (Mark)": "Environment-TTT (Mark)", "SFT + Short-Term Memory (Mark)": "SFT+Memory (Mark)", "Feedback-to-Policy TTT (Mark)": "F2P-TTT (Mark)"}
REMNANT_COLORS = {"Environment-prediction TTT (Remnant)": "#1f77b4", "SFT + Short-Term Memory (Remnant)": "#d62728", "Feedback-to-Policy TTT (Remnant)": "#2ca02c", "Lookahead Environment-TTT (Remnant)": "#9467bd"}
MARK_COLORS = {"Environment-prediction TTT (Mark)": "#1f77b4", "SFT + Short-Term Memory (Mark)": "#d62728", "Feedback-to-Policy TTT (Mark)": "#ff7f0e"}

def load(name):
    return [json.loads(x) for x in (RAW / name).read_text(encoding="utf-8").splitlines() if x.strip()]

def content_tokens(s):
    stop = {"the", "a", "an", "to", "of", "and", "or", "in", "on", "is", "are", "was", "were", "you", "your", "i", "my", "it", "this", "that", "with", "for", "from", "as", "at", "will", "can", "may", "be", "has", "have", "had", "not", "very", "then", "after", "before", "now", "next", "some", "there", "their", "they", "he", "she", "we", "me"}
    return {w for w in re.findall(r"[a-z][a-z0-9_'-]{2,}", (s or "").lower()) if w not in stop}

def extract_prediction(response):
    m = re.search(r'"predicted_outcome"\s*:\s*"((?:\\.|[^"\\])*)"', response or "")
    if not m:
        return ""
    try:
        return json.loads('"' + m.group(1) + '"')
    except Exception:
        return m.group(1)

def action_family(a):
    return (a or "").strip().split(" ", 1)[0].lower()

def state_fields(text):
    text = text or ""
    loc = re.search(r"Current Location:\s*([^\n]+)", text)
    health = re.search(r"My health is at\s*(\d+)", text)
    level = re.search(r"My level is\s*(\d+)", text)
    xp = re.search(r"My experience is at\s*(\d+)", text)
    return {"location": loc.group(1).strip() if loc else "", "health": int(health.group(1)) if health else np.nan, "level": int(level.group(1)) if level else np.nan, "experience": int(xp.group(1)) if xp else np.nan}

def failure_category(row, previous_action=None):
    obs = (row.get("observation") or {}).get("text", "")
    low = obs.lower()
    if row.get("invalid_action"):
        return "invalid action / parser failure"
    if (row.get("reward") or {}).get("death", 0) > 0:
        return "death / catastrophic transition"
    if re.search(r"nothing happens|no effect|doesn't work|does not work|cannot|can't|not enough|do not have|don't have|not allowed|failed", low):
        return "ineffective or impossible action"
    if previous_action and row.get("action") == previous_action:
        return "immediate action repetition"
    return ""

def build_frame(rows, method):
    rows = rows[:500]
    out, prev_actions = [], []
    for i, r in enumerate(rows):
        response = r.get("response", "") or ""
        obs = (r.get("observation") or {}).get("text", "")
        trace = r.get("lookahead_trace") or r.get("f2p_trace") or {}
        pred = trace.get("predicted_outcome") or trace.get("executed_action_prediction") or extract_prediction(response)
        prev_action = prev_actions[-1] if prev_actions else ""
        recent = prev_actions[-5:]
        reward = r.get("reward") or {}
        positive = sum(reward.get(k, 0) for k in ["unique_kill", "craft", "exploration", "quest", "side_quest"])
        failure = bool(failure_category(r, prev_action))
        history_ref = bool(re.search(r"\b(previous|earlier|before|last time|already|again|remember|recently|just)\b", response.lower()))
        no_progress_text = bool(re.search(r"nothing happens|no effect|doesn't work|does not work|cannot|can't|not enough|do not have|don't have", obs.lower()))
        pred_overlap = np.nan
        if pred:
            a, b = content_tokens(pred), content_tokens(obs)
            pred_overlap = len(a & b) / max(1, len(a | b))
        st = state_fields(obs)
        out.append({"method": method, "step": int(r.get("step", i)), "action": r.get("action", ""), "action_family": action_family(r.get("action", "")), "decision_time": float(r.get("decision_time", np.nan)), "input_tokens": float(r.get("num_input_tokens", np.nan)), "output_tokens": float(r.get("num_output_tokens", np.nan)), "response_chars": len(response), "response_words": len(response.split()), "invalid": int(bool(r.get("invalid_action"))), "failure": int(failure), "failure_category": failure_category(r, prev_action), "immediate_repeat": int(bool(prev_action and r.get("action") == prev_action)), "recent_repeat": int(bool(recent and r.get("action") in recent)), "positive_events": positive, "death": reward.get("death", 0), "quest": reward.get("quest", 0), "exploration": reward.get("exploration", 0), "craft": reward.get("craft", 0), "unique_kill": reward.get("unique_kill", 0), "kill": reward.get("kill", 0), "trade": reward.get("trade", 0), "side_quest": reward.get("side_quest", 0), "history_reference": int(history_ref), "no_progress_text": int(no_progress_text), "prediction_present": int(bool(pred)), "prediction_overlap": pred_overlap, "location": st["location"], "health": st["health"], "level": st["level"], "experience": st["experience"], "raw_response": response, "raw_observation": obs})
        prev_actions.append(r.get("action", ""))
    df = pd.DataFrame(out)
    for w in [10, 25, 50]:
        for col in ["invalid", "failure", "immediate_repeat", "recent_repeat", "positive_events", "no_progress_text", "prediction_present", "history_reference"]:
            df[f"{col}_roll{w}"] = df[col].rolling(w, min_periods=1).mean()
    ent, distinct, seen = [], [], set()
    for i in range(len(df)):
        window = df.iloc[max(0, i - 49): i + 1]["action"].tolist()
        c, n = Counter(window), len(window)
        ent.append(-sum((v / n) * math.log(v / n + 1e-12) for v in c.values()))
        seen.add(df.iloc[i]["action"]); distinct.append(len(seen))
    df["action_entropy_roll50"] = ent; df["unique_actions_cum"] = distinct
    df["positive_cum"] = df["positive_events"].cumsum(); df["death_cum"] = df["death"].cumsum(); df["quest_cum"] = df["quest"].cumsum(); df["exploration_cum"] = df["exploration"].cumsum()
    return df

ttt = build_frame(load("ttt_agent_log.jsonl"), "Environment-prediction TTT")
sft = build_frame(load("sft_agent_log.jsonl"), "SFT + Short-Term Memory")
f2p = build_frame(load("f2p_remnant_fair_agent_log.jsonl"), "Feedback-to-Policy TTT (Remnant)")
f2p_metro = build_frame(load("f2p_metropolis_fair_agent_log.jsonl"), "Feedback-to-Policy TTT (Metropolis)")
rem_ttt = build_frame(load("remnant_ttt_remote.jsonl"), "Environment-prediction TTT (Remnant)")
rem_sft = build_frame(load("remnant_sft_remote.jsonl"), "SFT + Short-Term Memory (Remnant)")
mark_ttt = build_frame(load("mark_ttt_agent_log.jsonl"), "Environment-prediction TTT (Mark)")
mark_sft = build_frame(load("mark_sft_agent_log.jsonl"), "SFT + Short-Term Memory (Mark)")
mark_f2p = build_frame(load("f2p_mark_fair_agent_log.jsonl"), "Feedback-to-Policy TTT (Mark)")
lookahead_remnant = build_frame(load("lookahead_remnant_agent_log.jsonl"), "Lookahead Environment-TTT (Remnant)")
lookahead_metropolis = build_frame(load("lookahead_metropolis_agent_log.jsonl"), "Lookahead Environment-TTT (Metropolis)")
hindsight_metropolis = build_frame(load("hindsight_metropolis_v1_agent_log.jsonl"), "Hindsight Long-Horizon TTT (Metropolis)")
# Main comparison is strictly within the same game: Metropolis only.
df = pd.concat([ttt, sft, f2p_metro, lookahead_metropolis, hindsight_metropolis], ignore_index=True)
df.to_json(OUT / "step_level_analysis.jsonl", orient="records", lines=True, force_ascii=False)

def plot_two(y, title, ylabel, filename, rolling=None, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for method, g in df.groupby("method", sort=False):
        z = g[y].to_numpy()
        if rolling: z = pd.Series(z).rolling(rolling, min_periods=1).mean().to_numpy()
        ax.plot(g["step"], z, label=LABELS[method], color=COLORS[method], linewidth=2)
    ax.set_title(title); ax.set_xlabel("Environment step"); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / filename, dpi=180); plt.close(fig)

def plot_components():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharex=True)
    for method, g in df.groupby("method", sort=False):
        for ax, col, title in zip(axes, ["quest_cum", "exploration_cum", "death_cum"], ["Quest progress", "Exploration progress", "Deaths"]):
            ax.plot(g.step, g[col], label=LABELS[method], color=COLORS[method], linewidth=2)
            ax.set_title(title); ax.set_xlabel("Step"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Cumulative events"); axes[-1].legend(frameon=False)
    fig.suptitle("Metropolis outcome trajectories (same seed 42)", y=1.02); fig.tight_layout(); fig.savefig(FIG / "01_outcome_trajectories.png", dpi=180); plt.close(fig)

plot_components()
plot_two("invalid_roll25", "Parser / invalid-action rate", "25-step rolling rate", "02_invalid_actions.png", ylim=(0, .35))
plot_two("recent_repeat_roll25", "Short-horizon action repetition", "25-step rolling rate", "03_action_repetition.png", ylim=(0, 1))
plot_two("positive_events", "Positive environment events per step", "Events at step", "04_positive_events.png")
plot_two("failure_roll25", "Long-horizon failure pressure", "25-step rolling failure rate", "05_failure_pressure.png", ylim=(0, 1))
plot_two("unique_actions_cum", "Action-space exploration", "Distinct exact actions so far", "06_action_exploration.png")
plot_two("action_entropy_roll50", "Action distribution entropy", "50-step Shannon entropy", "07_action_entropy.png")
plot_two("decision_time", "Decision latency", "Seconds per decision", "08_decision_latency.png", rolling=10)
plot_two("input_tokens", "Context growth", "Input tokens", "09_context_length.png", rolling=10)
plot_two("history_reference_roll25", "Explicit history / memory references", "25-step rolling rate", "10_history_references.png", ylim=(0, 1))
plot_two("prediction_present_roll25", "Explicit predicted-outcome field in model output", "25-step rolling rate (SFT has no field)", "11_outcome_prediction_coverage.png", ylim=(0, 1))
plot_two("prediction_overlap", "Predicted outcome vs. actual observation text overlap", "Content-token Jaccard (diagnostic)", "12_prediction_overlap.png", rolling=25, ylim=(0, 1))

def summarize(g):
    score = {c: int(g[c].sum()) for c in ["unique_kill", "kill", "craft", "exploration", "death", "trade", "quest", "side_quest"]}
    positive_total = sum(score[c] for c in ["unique_kill", "kill", "craft", "exploration", "trade", "quest", "side_quest"])
    return {"method": g.method.iloc[0], "steps": len(g), "invalid_actions": int(g.invalid.sum()), "invalid_rate": float(g.invalid.mean()), "failure_events": int(g.failure.sum()), "failure_rate": float(g.failure.mean()), "immediate_repeat_rate": float(g.immediate_repeat.mean()), "recent_repeat_rate": float(g.recent_repeat.mean()), "distinct_actions": int(g.action.nunique()), "mean_decision_seconds": float(g.decision_time.mean()), "median_decision_seconds": float(g.decision_time.median()), "mean_input_tokens": float(g.input_tokens.mean()), "mean_response_words": float(g.response_words.mean()), "history_reference_rate": float(g.history_reference.mean()), "prediction_coverage": float(g.prediction_present.mean()), "prediction_overlap_mean": float(g.prediction_overlap.mean(skipna=True)), "positive_events": int(g.positive_events.sum()), "positive_reward_total": positive_total, "raw_reward_total_including_death": positive_total + score["death"], "net_reward_proxy_positive_minus_death": positive_total - score["death"], "score_components": json.dumps(score, ensure_ascii=False)}

summary = pd.DataFrame([summarize(ttt), summarize(sft), summarize(f2p_metro), summarize(lookahead_metropolis), summarize(hindsight_metropolis)])
summary.to_csv(OUT / "summary_metrics.csv", index=False, encoding="utf-8-sig")
errors = df[df.failure == 1][["method", "step", "action", "failure_category", "invalid", "death", "raw_response", "raw_observation"]].copy()
errors.to_json(OUT / "failure_cases.jsonl", orient="records", lines=True, force_ascii=False)
cat = df[df.failure_category != ""].groupby(["method", "failure_category"]).size().reset_index(name="count")
cat.to_csv(OUT / "error_category_counts.csv", index=False, encoding="utf-8-sig")

fig, ax = plt.subplots(figsize=(11, 5.5))
pivot = cat.pivot(index="failure_category", columns="method", values="count").fillna(0)
pivot = pivot.reindex(columns=["Environment-prediction TTT", "SFT + Short-Term Memory", "Feedback-to-Policy TTT (Metropolis)", "Lookahead Environment-TTT (Metropolis)", "Hindsight Long-Horizon TTT (Metropolis)"], fill_value=0)
pivot.index = [x.replace(" / ", " /\n") for x in pivot.index]
pivot.plot(kind="bar", ax=ax, color=[COLORS[x] for x in pivot.columns])
ax.set_title("Long-horizon error categories"); ax.set_xlabel(""); ax.set_ylabel("Count"); ax.grid(axis="y", alpha=.25); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "13_error_categories.png", dpi=180); plt.close(fig)

def example_rows(g, n=5):
    return g[g.failure == 1].head(n)[["step", "action", "failure_category", "raw_response", "raw_observation"]].to_dict("records")

examples = {"ttt": example_rows(ttt), "sft": example_rows(sft), "f2p_metropolis": example_rows(f2p_metro), "lookahead_metropolis": example_rows(lookahead_metropolis), "hindsight_metropolis": example_rows(hindsight_metropolis)}
(OUT / "representative_examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")

# Remnant is kept as a separate environment and is never mixed into the Metropolis comparison figures or tables.
remnant_summary = pd.DataFrame([summarize(rem_ttt), summarize(rem_sft), summarize(f2p), summarize(lookahead_remnant)])
lookahead_remnant_summary = pd.DataFrame([summarize(lookahead_remnant)])
lookahead_metropolis_summary = pd.DataFrame([summarize(lookahead_metropolis)])
lookahead_remnant_summary.to_csv(OUT / "summary_lookahead_remnant.csv", index=False, encoding="utf-8-sig")
lookahead_metropolis_summary.to_csv(OUT / "summary_lookahead_metropolis.csv", index=False, encoding="utf-8-sig")

def lookahead_diag(g):
    traces = [r.get("lookahead_trace") or {} for r in load("lookahead_remnant_agent_log.jsonl" if "Remnant" in g.method.iloc[0] else "lookahead_metropolis_agent_log.jsonl")]
    n = len(traces)
    return {"method": g.method.iloc[0], "steps": n,
            "initial_prediction_coverage": sum(bool(t.get("initial_predicted_outcome")) for t in traces) / max(1, n),
            "revision_prediction_coverage": sum(bool(t.get("revised_predicted_outcome")) for t in traces) / max(1, n),
            "executed_prediction_coverage": sum(bool(t.get("executed_action_prediction")) for t in traces) / max(1, n),
            "initial_to_revision_change_rate": sum(t.get("initial_action") != t.get("revised_action") for t in traces) / max(1, n),
            "training_trigger_rate": sum(bool(t.get("training_triggered")) for t in traces) / max(1, n),
            "training_updates": sum(max(0, (t.get("training_steps_total_after") or 0) - (t.get("training_steps_total_before") or 0)) for t in traces)}

lookahead_diagnostics = pd.DataFrame([lookahead_diag(lookahead_remnant), lookahead_diag(lookahead_metropolis)])
lookahead_diagnostics.iloc[[0]].to_csv(OUT / "lookahead_diagnostics_remnant.csv", index=False, encoding="utf-8-sig")
lookahead_diagnostics.iloc[[1]].to_csv(OUT / "lookahead_diagnostics_metropolis.csv", index=False, encoding="utf-8-sig")
remnant_summary.to_csv(OUT / "summary_remnant.csv", index=False, encoding="utf-8-sig")
(OUT / "representative_examples_remnant.json").write_text(json.dumps({"f2p_remnant": example_rows(f2p), "lookahead_remnant": example_rows(lookahead_remnant)}, ensure_ascii=False, indent=2), encoding="utf-8")
mark_summary = pd.DataFrame([summarize(mark_ttt), summarize(mark_sft), summarize(mark_f2p)])
mark_summary.to_csv(OUT / "summary_mark.csv", index=False, encoding="utf-8-sig")
(OUT / "representative_examples_mark.json").write_text(json.dumps({"ttt": example_rows(mark_ttt), "sft": example_rows(mark_sft), "f2p": example_rows(mark_f2p)}, ensure_ascii=False, indent=2), encoding="utf-8")
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
axes[0].plot(f2p.step, f2p.failure_roll25, color=COLORS["Feedback-to-Policy TTT (Remnant)"])
axes[0].set_title("Remnant failure pressure"); axes[0].set_xlabel("Step"); axes[0].set_ylabel("25-step rate")
axes[1].plot(f2p.step, f2p.recent_repeat_roll25, color=COLORS["Feedback-to-Policy TTT (Remnant)"])
axes[1].set_title("Remnant action repetition"); axes[1].set_xlabel("Step"); axes[1].set_ylabel("25-step rate")
axes[2].plot(f2p.step, f2p.prediction_present_roll25, color=COLORS["Feedback-to-Policy TTT (Remnant)"])
axes[2].set_title("Remnant prediction coverage"); axes[2].set_xlabel("Step"); axes[2].set_ylabel("25-step rate")
for ax in axes: ax.grid(alpha=.25)
fig.suptitle("Remnant-only F2P diagnostics", y=1.02); fig.tight_layout(); fig.savefig(FIG / "remnant_diagnostics.png", dpi=180); plt.close(fig)

# Remnant comparison figures are also strictly within Remnant; no Metropolis curve is included.
remnant_df = pd.concat([rem_ttt, rem_sft, f2p, lookahead_remnant], ignore_index=True)
def plot_remnant(y, title, ylabel, filename, rolling=None, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for method, g in remnant_df.groupby("method", sort=False):
        z = g[y].to_numpy()
        if rolling: z = pd.Series(z).rolling(rolling, min_periods=1).mean().to_numpy()
        ax.plot(g["step"], z, label=LABELS[method], color=REMNANT_COLORS[method], linewidth=2)
    ax.set_title(title); ax.set_xlabel("Environment step"); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / filename, dpi=180); plt.close(fig)

plot_remnant("failure_roll25", "Remnant failure pressure", "25-step rolling failure rate", "remnant_01_failure_pressure.png", ylim=(0, 1))
plot_remnant("recent_repeat_roll25", "Remnant action repetition", "25-step rolling rate", "remnant_02_action_repetition.png", ylim=(0, 1))
plot_remnant("unique_actions_cum", "Remnant action-space exploration", "Distinct exact actions so far", "remnant_03_action_exploration.png")
plot_remnant("invalid_roll25", "Remnant invalid-action rate", "25-step rolling rate", "remnant_04_invalid_actions.png", ylim=(0, .8))
plot_remnant("prediction_present_roll25", "Remnant predicted-outcome coverage", "25-step rolling rate", "remnant_05_prediction_coverage.png", ylim=(0, 1))

# Mark comparison figures are strictly within Mark; no other game is included.
mark_df = pd.concat([mark_ttt, mark_sft, mark_f2p], ignore_index=True)
def plot_mark(y, title, ylabel, filename, rolling=None, ylim=None):
    fig, ax = plt.subplots(figsize=(10, 5.4))
    for method, g in mark_df.groupby("method", sort=False):
        z = g[y].to_numpy()
        if rolling: z = pd.Series(z).rolling(rolling, min_periods=1).mean().to_numpy()
        ax.plot(g["step"], z, label=LABELS[method], color=MARK_COLORS[method], linewidth=2)
    ax.set_title(title); ax.set_xlabel("Environment step"); ax.set_ylabel(ylabel)
    if ylim: ax.set_ylim(*ylim)
    ax.grid(alpha=.25); ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG / filename, dpi=180); plt.close(fig)

plot_mark("failure_roll25", "Mark failure pressure", "25-step rolling failure rate", "mark_01_failure_pressure.png", ylim=(0, 1))
plot_mark("recent_repeat_roll25", "Mark action repetition", "25-step rolling rate", "mark_02_action_repetition.png", ylim=(0, 1))
plot_mark("unique_actions_cum", "Mark action-space exploration", "Distinct exact actions so far", "mark_03_action_exploration.png")
plot_mark("invalid_roll25", "Mark invalid-action rate", "25-step rolling rate", "mark_04_invalid_actions.png", ylim=(0, .8))
plot_mark("prediction_present_roll25", "Mark predicted-outcome coverage", "25-step rolling rate", "mark_05_prediction_coverage.png", ylim=(0, 1))

def plot_lookahead_diag(row, filename, title, color):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["Initial\nprediction", "Revision\nprediction", "Executed\naction prediction"]
    vals = [row.initial_prediction_coverage, row.revision_prediction_coverage, row.executed_prediction_coverage]
    axes[0].bar(labels, vals, color=color); axes[0].set_ylim(0, 1); axes[0].set_ylabel("Coverage"); axes[0].set_title("Prediction stages")
    axes[1].bar(["Action\nchanged", "Training\ntriggered"], [row.initial_to_revision_change_rate, row.training_trigger_rate], color=color)
    axes[1].set_ylim(0, 1); axes[1].set_ylabel("Rate"); axes[1].set_title("Lookahead intervention")
    fig.suptitle(title, fontsize=13, weight="bold");
    for ax in axes: ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(FIG / filename, dpi=190); plt.close(fig)

plot_lookahead_diag(lookahead_diagnostics.iloc[0], "lookahead_remnant_diagnostics.png", "Remnant — Lookahead Environment-TTT diagnostics", "#9467bd")
plot_lookahead_diag(lookahead_diagnostics.iloc[1], "lookahead_metropolis_diagnostics.png", "Metropolis — Lookahead Environment-TTT diagnostics", "#8c564b")

# Results overview: separate panels and tables for each game; no cross-game score comparison.
def overview_table(frame, title, color):
    rows = []
    for _, r in frame.iterrows():
        score = json.loads(r["score_components"])
        rows.append([LABELS.get(r["method"], r["method"]), score["unique_kill"], score["kill"], score["craft"], score["exploration"], score["death"], score["trade"], score["quest"], score["side_quest"], int(r["positive_reward_total"]), int(r["raw_reward_total_including_death"]), int(r["net_reward_proxy_positive_minus_death"])])
    return rows

fig, axes = plt.subplots(3, 1, figsize=(16, 13.5), gridspec_kw={"height_ratios": [1.25, 1, 1]})
headers = ["Method", "UKill", "Kill", "Craft", "Explore", "Death", "Trade", "Quest", "Side quest", "Positive total", "Raw total", "Net proxy"]
for ax, frame, title, color in [(axes[0], summary, "Metropolis — same-task results", "#ff7f0e"), (axes[1], remnant_summary, "Remnant — same-task results", "#2ca02c"), (axes[2], mark_summary, "Mark — same-task results", "#7f3fbf")]:
    ax.axis("off")
    table = ax.table(cellText=overview_table(frame, title, color), colLabels=headers, cellLoc="center", loc="center", colWidths=[.20, .065, .055, .065, .075, .065, .055, .065, .085, .105, .085, .09])
    table.auto_set_font_size(False); table.set_fontsize(9); table.scale(1, 2.0)
    for col in range(len(headers)):
        cell = table[(0, col)]; cell.set_facecolor(color); cell.set_text_props(color="white", weight="bold")
    ax.set_title(title, fontsize=14, weight="bold", pad=12)
fig.suptitle("AgentOdyssey Results Overview — scores are separated by task", fontsize=16, weight="bold", y=.98)
fig.text(.01, .01, "Positive total = unique_kill + kill + craft + exploration + trade + quest + side_quest; Net proxy = positive total − death. These are logged reward-component sums, not a replacement for an official benchmark aggregate if the benchmark defines another weighting.", fontsize=8)
fig.tight_layout(rect=[0, .04, 1, .95]); fig.savefig(FIG / "00_results_overview.png", dpi=220); plt.close(fig)
print(summary.to_string(index=False)); print("Generated", len(list(FIG.glob("*.png"))), "figures")
