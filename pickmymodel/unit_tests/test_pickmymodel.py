"""Tests that need no Ollama: everything uses a mocked client.

Run with:  python -m pytest unit_tests
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import benchmark as bm            # noqa: E402
import recommend as rec           # noqa: E402
import scoring as sc              # noqa: E402


# ---------------------------------------------------------------- mocked Ollama client
class FakeClient:
    """Mimics ollama.Client. `answers` maps model name -> reply text."""

    def __init__(self, answers=None, fail_models=(), support_think=True):
        self.answers = answers or {}
        self.fail_models = fail_models
        self.support_think = support_think
        self.unloaded = []
        self.calls = []

    def chat(self, model, messages, options=None, **kwargs):
        if "think" in kwargs and not self.support_think:
            raise TypeError("unexpected keyword argument 'think'")
        self.calls.append(model)
        if model in self.fail_models:
            raise RuntimeError("boom")
        text = self.answers.get(model, "OK")
        user = messages[-1]["content"]
        if model == bm.JUDGE_MODEL:
            text = "4"
        elif callable(text):
            text = text(user)
        return {"message": {"content": text}, "eval_count": 20, "eval_duration": 1_000_000_000}

    def generate(self, model, prompt="", keep_alive=None):
        self.unloaded.append(model)
        return {}

    def list(self):
        return {"models": [{"name": "a:1b"}, {"name": "nomic-embed-text"}, {"name": "b:2b"}]}


def smart(user):
    """Reply that passes the built-in tests."""
    u = user.lower()
    if "is_even" in u:
        return "def is_even(n):\n    return n % 2 == 0"
    if "len(" in u:
        return "3"
    return "OK"


# ---------------------------------------------------------------- scoring
def test_strip_think():
    assert sc.strip_think("<think>hmm</think>\nAnswer") == "Answer"
    assert sc.strip_think("<think>never closed") == ""


def test_json_valid_fences_and_prose():
    t = {"check": "json_valid", "expected": ["name", "age"]}
    assert sc.score_answer(t, 'Sure!\n```json\n{"name": "A", "age": 3}\n```\nDone')
    assert sc.score_answer(t, 'Here: {"name": "A", "age": 3} hope it helps')
    assert not sc.score_answer(t, '{"name": "A"}')
    assert not sc.score_answer(t, "no json here")
    assert sc.score_answer({"check": "json_valid", "expected": []}, "[1, 2]")


def test_keywords_case_and_min_matches():
    t = {"check": "contains_keywords", "expected": ["Solar", "bill"], "min_matches": 1}
    assert sc.score_answer(t, "SOLAR panels")
    t["min_matches"] = 2
    assert not sc.score_answer(t, "solar only")
    assert sc.score_answer({"check": "contains_keywords", "expected": ["a", "b"]}, "A and B")


def test_exact_match_ignores_case_quotes_punctuation():
    t = {"check": "exact_match", "expected": "Hello"}
    for ans in ["hello", '"Hello"', "HELLO.", "hello!", "  'hello'  "]:
        assert sc.score_answer(t, ans), ans
    assert not sc.score_answer(t, "hello world")


def test_tool_call():
    t = {"check": "tool_call", "expected": {"tool": "get_weather", "arguments": ["city"]}}
    assert sc.score_answer(t, '{"tool": "get_weather", "arguments": {"city": "Pune"}}')
    assert not sc.score_answer(t, '{"tool": "get_weather", "arguments": {}}')
    assert not sc.score_answer(t, '{"tool": "other", "arguments": {"city": "x"}}')


def test_unknown_check_raises():
    with pytest.raises(ValueError):
        sc.score_answer({"check": "nope"}, "x")


# ---------------------------------------------------------------- tests files
def test_builtin_tests_load():
    tests = bm.load_tests()
    by_task = {}
    for t in tests:
        by_task.setdefault(t["task_type"], []).append(t)
    for task in ("coding", "summarizing", "extraction"):
        assert len(by_task[task]) == 5
    assert len({t["id"] for t in tests}) == len(tests)
    assert [t["id"] for t in bm.load_tests(["extraction"])] == [t["id"] for t in by_task["extraction"]]


def test_bad_test_file(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps([{"id": "x"}]))
    with pytest.raises(ValueError):
        bm.load_tests(tests_dir=str(tmp_path))


# ---------------------------------------------------------------- model listing
def test_list_models_new_and_old_formats():
    assert bm.list_models(FakeClient()) == ["a:1b", "b:2b"]          # old dict format, embed skipped

    class M:  # new library: objects with .model
        def __init__(self, model):
            self.model = model

    class New:
        def list(self):
            return type("R", (), {"models": [M("x:1"), M("my-embed"), M("y:2")]})()

    assert bm.list_models(New()) == ["x:1", "y:2"]


def test_ollama_status_when_down():
    class Down:
        def list(self):
            raise ConnectionError("refused")

    models, error = bm.ollama_status(Down())
    assert models == [] and "not running" in error


# ---------------------------------------------------------------- recommendations
def _res(model, task, passed, tps):
    return {"model": model, "task_type": task, "passed": passed, "tokens_per_sec": tps,
            "time_s": 1.0, "error": None}


def test_recommend_fastest_above_threshold():
    results = ([_res("slow", "coding", True, 5.0)] * 5 + [_res("fast", "coding", True, 20.0)] * 4
               + [_res("fast", "coding", False, 20.0)])
    pick = rec.recommend(results)["coding"]
    assert pick["model"] == "fast"                                   # 80% >= 70% and faster
    assert "Best for coding on this machine: fast (80% pass rate, 20.0 tokens/sec)" in \
        rec.recommendation_sentences(results)


def test_recommend_falls_back_to_highest_pass_rate():
    results = ([_res("a", "coding", True, 5.0)] * 6 + [_res("a", "coding", False, 5.0)] * 4
               + [_res("b", "coding", True, 50.0)] * 3 + [_res("b", "coding", False, 50.0)] * 7)
    assert rec.recommend(results)["coding"]["model"] == "a"          # nobody reaches 70%


def test_guess_task_type():
    assert rec.guess_task_type("Fix this Python function bug") == "coding"
    assert rec.guess_task_type("Please summarize this article") == "summarizing"
    assert rec.guess_task_type("Extract the email address as JSON") == "extraction"
    assert rec.guess_task_type("hello there") is None


def test_ask_winner_rules():
    rows = [{"model": "a", "judge_score": 5, "tokens_per_sec": 5, "error": None},
            {"model": "b", "judge_score": 4, "tokens_per_sec": 15, "error": None},
            {"model": "c", "judge_score": 2, "tokens_per_sec": 99, "error": None}]
    assert rec.pick_ask_winner(rows)[0] == "b"                       # fastest with score >= 4
    rows[0]["judge_score"], rows[1]["judge_score"] = 3, 2
    assert rec.pick_ask_winner(rows)[0] == "a"                       # highest score
    for r in rows:
        r["judge_score"] = None
    winner, verdict = rec.pick_ask_winner(rows)
    assert winner == "c" and "judge" in verdict.lower()
    assert rec.pick_ask_winner([{"model": "x", "error": "e"}])[0] is None


# ---------------------------------------------------------------- benchmark run
def _run(client, **kw):
    tests = bm.load_tests(["coding", "extraction"])
    return bm.run_benchmark(["m1", "m2"], tests, client_factory=lambda t: client, **kw), tests


def test_benchmark_run_with_mock(tmp_path):
    client = FakeClient({"m1": smart, "m2": "nonsense"}, fail_models=())
    run, tests = _run(client)
    assert len(run["results"]) == 2 * len(tests)
    m1 = [r for r in run["results"] if r["model"] == "m1"]
    assert any(r["passed"] for r in m1) and all(r["tokens_per_sec"] == 20.0 for r in m1)
    assert set(client.unloaded) >= {"m1", "m2"}                      # unloaded after each model
    assert "load_time_s" in run["models"]["m1"] and "peak_ram_gb" in run["models"]["m1"]
    path = bm.save_run(run, str(tmp_path))
    assert os.path.basename(path).startswith("run_") and json.load(open(path))["results"]
    lb = bm.leaderboard(run)
    assert {"Model", "Avg tokens/sec", "Load time (s)", "Peak RAM (GB)", "Pass rate (%)"} <= set(lb.columns)


def test_errors_never_stop_the_run():
    client = FakeClient({"m1": smart}, fail_models=("m2",))
    run, tests = _run(client)
    m2 = [r for r in run["results"] if r["model"] == "m2"]
    assert len(m2) == len(tests) and all(r["error"] and not r["passed"] for r in m2)
    assert any(r["passed"] for r in run["results"] if r["model"] == "m1")


def test_think_fallback_for_old_library():
    client = FakeClient({"m1": smart}, support_think=False)
    run, _ = _run(client)
    assert any(r["passed"] for r in run["results"] if r["model"] == "m1")


def test_repeats_majority_and_progress():
    seen = []
    tests = bm.load_tests(["coding"])[:1]
    outputs = iter(["wrong", "def is_even(n): return True", "def is_even(n): return True"])
    client = FakeClient({"m1": lambda u: next(outputs) if "is_even" in u else "x"})
    run = bm.run_benchmark(["m1"], tests, repeats=3, progress=lambda d, t, l: seen.append((d, t)),
                           client_factory=lambda t: client)
    # warm-up call uses "Say OK." (not is_even), so the three repeats get wrong / right / right
    r = run["results"][0]
    assert r["repeats"] == 3 and r["passes"] == 2 and r["passed"] is True
    assert seen[-1] == (1, 1)


def test_judge_second_phase_and_ram_warning():
    client = FakeClient({"m1": smart, "m2": "x"})
    run, _ = _run(client, judge=True)
    assert run["judge"] and all(r["judge_score"] == 4 for r in run["results"] if r["answer"])
    # judge model calls all happen after every benchmarked model call (never together in RAM)
    first_judge = client.calls.index(bm.JUDGE_MODEL)
    assert all(m == bm.JUDGE_MODEL for m in client.calls[first_judge:])
    assert "Judge score (1-5)" in bm.leaderboard(run).columns
    run["models"]["m1"]["peak_ram_gb"] = run["system"]["ram_total_gb"] * 0.95
    assert "WARNING" in bm.leaderboard(run).set_index("Model").loc["m1", "Warning"]


def test_parse_score_and_reference_averaging():
    assert bm.parse_score("<think>x</think> Score: 5") == 5
    assert bm.parse_score("none") is None
    score, err = bm.judge_one(FakeClient(), "q", "a", reference="ref")
    assert score == 4.0 and err is None


# ---------------------------------------------------------------- Ask / router / exports
def test_ask_models_and_saved_compare(tmp_path):
    client = FakeClient({"a": "answer a", "b": "answer b"})
    res = bm.ask_models("Summarize this text please", ["a", "b"], judge=True, reference="ref",
                        client_factory=lambda t: client, results_dir=str(tmp_path))
    assert res["winner"] in ("a", "b") and os.path.basename(res["path"]).startswith("compare_")
    assert res["reference_used"] and res["task_guess"] == "summarizing"
    assert json.load(open(res["path"]))["rows"][0]["judge_score"] == 4.0


def test_router_uses_latest_benchmark_and_fallbacks(tmp_path):
    prompt = "Fix this python function"
    assert rec.choose_router_model(prompt, ["a", "b"], str(tmp_path))[0] == "a"   # no benchmark yet
    run = {"results": [_res("fast", "coding", True, 30.0), _res("slow", "coding", True, 5.0),
                       _res("fast", "extraction", True, 30.0)]}
    json.dump(run, open(tmp_path / "run_1.json", "w"))
    model, reason, task = rec.choose_router_model(prompt, ["fast", "slow"], str(tmp_path))
    assert (model, task) == ("fast", "coding") and "benchmark" in reason.lower()
    assert "coding task" in rec.hint_from_latest(prompt, str(tmp_path))
    client = FakeClient({"fast": "done"})
    out = bm.auto_answer(prompt, ["fast", "slow"], str(tmp_path), client_factory=lambda t: client)
    assert out["model"] == "fast" and out["answer"] == "done"


def test_history_compare_and_anonymous_profile():
    run_a, _ = _run(FakeClient({"m1": smart, "m2": "x"}))
    run_b, _ = _run(FakeClient({"m1": smart, "m2": "x"}))
    table = bm.compare_runs(run_a, run_b)
    assert set(table["Model"]) == {"m1", "m2"} and "tokens/sec change" in table.columns
    text = json.dumps(bm.anonymous_profile(run_a))
    assert "m1" in text and "is_even" not in text and "Say OK" not in text
    for banned in ("prompt", "answer", "path", "user"):
        assert banned not in json.loads(text).keys()
    assert bm.df_to_markdown(bm.leaderboard(run_a)).startswith("| Model")


def test_gemma_helpers():
    class RouterClient(FakeClient):
        def chat(self, model, messages, options=None, **kw):
            return {"message": {"content": "Coding"}, "eval_count": 1, "eval_duration": 1}

    assert bm.gemma_route_task("anything", lambda t: RouterClient()) == "coding"
    run, _ = _run(FakeClient({"m1": smart}))
    assert isinstance(bm.gemma_explain(run, lambda t: RouterClient()), str)
