"""recommend.py - recommendation rules and prompt-routing helpers.

No Ollama calls here, so everything in this file is easy to test.
A "result" is one dict per (model, test) with at least these keys:
    model, task_type, passed, time_s, tokens_per_sec, error
"""
import glob
import json
import os

PASS_THRESHOLD = 0.70  # a model needs at least 70% pass rate to be "good enough"

# Keywords used to guess what kind of task a pasted prompt is.
TASK_KEYWORDS = {
    "coding": ["code", "function", "python", "javascript", "bug", "script", "class ",
               "def ", "program", "algorithm", "sql", "regex", "compile", "debug"],
    "summarizing": ["summarize", "summarise", "summary", "tl;dr", "tldr", "shorten",
                    "key points", "in one sentence", "condense"],
    "extraction": ["extract", "json", "parse", "pull out", "list all", "fields",
                   "email address", "phone number", "from this text", "structured"],
    "tool_calling": ["tool", "function call", "api call", "call the", "arguments"],
}


def guess_task_type(prompt):
    """Guess the task type from keywords. Returns None when nothing matches."""
    text = prompt.lower()
    scores = {task: sum(1 for k in words if k in text) for task, words in TASK_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def _avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else 0.0


def task_stats(results):
    """Return {task_type: {model: {"pass_rate": 0-1, "tps": avg tokens/sec}}}."""
    grouped = {}
    for r in results:
        grouped.setdefault(r["task_type"], {}).setdefault(r["model"], []).append(r)
    stats = {}
    for task, models in grouped.items():
        stats[task] = {}
        for model, rows in models.items():
            passed = sum(1 for r in rows if r.get("passed"))
            tps = _avg([r.get("tokens_per_sec") for r in rows if r.get("tokens_per_sec")])
            stats[task][model] = {"pass_rate": passed / len(rows), "tps": tps}
    return stats


def recommend(results):
    """Apply the rule per task type.

    1. Fastest model (highest tokens/sec) whose pass rate is >= 70%.
    2. If nobody reaches 70%, the highest pass rate (speed breaks ties).
    Returns {task_type: {"model", "pass_rate", "tps"}}.
    """
    picks = {}
    for task, models in task_stats(results).items():
        good = {m: s for m, s in models.items() if s["pass_rate"] >= PASS_THRESHOLD}
        if good:
            best = max(good, key=lambda m: good[m]["tps"])
        else:
            best = max(models, key=lambda m: (models[m]["pass_rate"], models[m]["tps"]))
        picks[task] = {"model": best, **models[best]}
    return picks


def recommendation_sentences(results):
    """Plain sentences like: Best for coding on this machine: qwen3:4b (100% pass rate, 19.3 tokens/sec)."""
    lines = []
    for task, p in sorted(recommend(results).items()):
        label = task.replace("_", " ")
        lines.append("Best for %s on this machine: %s (%d%% pass rate, %.1f tokens/sec)"
                     % (label, p["model"], round(p["pass_rate"] * 100), p["tps"]))
    return lines


def pick_ask_winner(rows):
    """Choose the winner of an Ask-tab comparison.

    rows: list of dicts with model, judge_score (int or None), time_s, tokens_per_sec, error.
    Returns (winner_model_or_None, one_sentence_verdict).
    """
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return None, "No model produced an answer. Check the errors in the table."
    scored = [r for r in ok if r.get("judge_score") is not None]
    if scored:
        strong = [r for r in scored if r["judge_score"] >= 4]
        if strong:
            w = max(strong, key=lambda r: r.get("tokens_per_sec") or 0)
            return w["model"], ("%s is the best pick: judge score %s/5 and the fastest strong answer (%.1f tokens/sec)."
                                % (w["model"], w["judge_score"], w.get("tokens_per_sec") or 0))
        w = max(scored, key=lambda r: (r["judge_score"], r.get("tokens_per_sec") or 0))
        return w["model"], ("%s scored highest (%s/5), but no model reached 4/5, so consider a larger model."
                            % (w["model"], w["judge_score"]))
    w = max(ok, key=lambda r: r.get("tokens_per_sec") or 0)
    return w["model"], ("%s was the fastest (%.1f tokens/sec). Turn on the judge to also compare answer quality."
                        % (w["model"], w.get("tokens_per_sec") or 0))


# ---------- saved-run helpers ----------

def list_run_files(results_dir, prefix="run_"):
    """Saved result files, oldest first."""
    return sorted(glob.glob(os.path.join(results_dir, prefix + "*.json")))


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def latest_run(results_dir):
    """Load the newest saved benchmark run, or None."""
    files = list_run_files(results_dir)
    return load_json(files[-1]) if files else None


def hint_from_latest(prompt, results_dir):
    """Instant hint for the Ask tab, taken from the latest saved benchmark (no model call)."""
    task = guess_task_type(prompt)
    if task is None:
        return "Could not guess a task type from your prompt, so no instant hint is available."
    run = latest_run(results_dir)
    if not run:
        return "This looks like a %s task. Run a benchmark first to get an instant hint." % task.replace("_", " ")
    picks = recommend(run["results"])
    if task not in picks:
        return "This looks like a %s task, but the latest benchmark did not include it." % task.replace("_", " ")
    p = picks[task]
    return ("This looks like a %s task. Your latest benchmark suggests %s (%d%% pass rate, %.1f tokens/sec)."
            % (task.replace("_", " "), p["model"], round(p["pass_rate"] * 100), p["tps"]))


def choose_router_model(prompt, installed, results_dir, task=None):
    """Pick the model for 'Auto-answer'. Returns (model, reason, task).

    Uses the latest benchmark recommendation for the task type; if none exists,
    falls back to the fastest installed model (from any run), or the first installed one.
    """
    task = task or guess_task_type(prompt)
    run = latest_run(results_dir)
    if run:
        picks = recommend(run["results"])
        if task in picks and picks[task]["model"] in installed:
            p = picks[task]
            return p["model"], ("Latest benchmark: best model for %s (%d%% pass rate, %.1f tokens/sec)."
                                % (task.replace("_", " "), round(p["pass_rate"] * 100), p["tps"])), task
        # no task-specific pick: fastest installed model across the whole run
        speeds = {}
        for r in run["results"]:
            if r["model"] in installed and not r.get("error"):
                speeds.setdefault(r["model"], []).append(r.get("tokens_per_sec") or 0)
        if speeds:
            m = max(speeds, key=lambda k: _avg(speeds[k]))
            return m, "No recommendation for this task type, so using the fastest benchmarked model.", task
    if installed:
        return installed[0], "No saved benchmark found, so using the first installed model.", task
    return None, "No models are installed.", task
