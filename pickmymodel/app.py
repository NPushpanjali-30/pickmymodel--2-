"""app.py - Streamlit dashboard for PickMyModel.

Start it with:   streamlit run app.py
All heavy lifting lives in benchmark.py / recommend.py; this file is only the user interface.
"""
import json
import os

import pandas as pd
import streamlit as st

from benchmark import (JUDGE_MODEL, RESULTS_DIR, anonymous_profile, ask_models, auto_answer,
                       compare_runs, df_to_markdown, gemma_explain, history_tokens_per_sec,
                       leaderboard, load_tests, ollama_status, run_benchmark, save_run,
                       system_info)
from recommend import hint_from_latest, list_run_files, load_json, recommendation_sentences

st.set_page_config(page_title="PickMyModel", page_icon="🎯", layout="wide")
st.title("🎯 PickMyModel")
st.caption("Benchmark the open-weight models on your own laptop and pick the best one per task. "
           "Private, free and offline.")

# ---------------------------------------------------------------- header: hardware
info = system_info()
c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
c1.metric("RAM total", "%.1f GB" % info["ram_total_gb"])
c2.metric("RAM free", "%.1f GB" % info["ram_free_gb"])
c3.metric("CPU cores / threads", "%s / %s" % (info["physical_cores"], info["threads"]))
c4.metric("CPU", info["cpu_name"])

# ---------------------------------------------------------------- sidebar
installed, ollama_error = ollama_status()
st.sidebar.header("Benchmark settings")
if ollama_error:
    st.sidebar.error(ollama_error)
elif not installed:
    st.sidebar.warning("Ollama is running but no models are installed. Try: `ollama pull qwen3:4b`")

try:
    all_tests = load_tests()
    task_options = sorted({t["task_type"] for t in all_tests})
    tests_error = None
except ValueError as exc:
    all_tests, task_options, tests_error = [], [], str(exc)
    st.sidebar.error("Test files problem: " + tests_error)

sel_models = st.sidebar.multiselect("Models", installed, default=installed)
sel_tasks = st.sidebar.multiselect("Task types", task_options, default=task_options)
use_judge = st.sidebar.checkbox("Judge answers with %s" % JUDGE_MODEL, value=False,
                                help="Optional second phase. Slower, and the small judge model can be biased.")
timeout = st.sidebar.number_input("Timeout per call (seconds)", 10, 600, 60, step=10)
repeats = st.sidebar.number_input("Repeats per test", 1, 10, 1)
run_clicked = st.sidebar.button("Run benchmark", type="primary",
                                disabled=bool(ollama_error or not installed or tests_error))
progress_slot = st.sidebar.empty()

# Older results selector
run_files = list_run_files(RESULTS_DIR)
older_choice = st.sidebar.selectbox("Load older results", ["(current / latest)"] + [os.path.basename(f) for f in reversed(run_files)])

if run_clicked:
    if not sel_models or not sel_tasks:
        st.sidebar.warning("Pick at least one model and one task type.")
    else:
        bar = progress_slot.progress(0.0, text="Starting...")

        def show_progress(done, total, label):
            bar.progress(done / max(total, 1), text="[%d/%d] %s" % (done, total, label))

        run = run_benchmark(sel_models, load_tests(sel_tasks), use_judge, int(timeout), None,
                            int(repeats), show_progress)
        st.session_state["run"] = run
        st.session_state["run_path"] = save_run(run)
        st.session_state["explanation"] = None
        progress_slot.success("Done. Saved to results/")
        run_files = list_run_files(RESULTS_DIR)

# Decide which run to display
run = None
if older_choice != "(current / latest)":
    run = load_json(os.path.join(RESULTS_DIR, older_choice))
elif "run" in st.session_state:
    run = st.session_state["run"]
elif run_files:
    run = load_json(run_files[-1])

tab_bench, tab_ask = st.tabs(["Benchmark", "Ask"])

# ================================================================ TAB 1: Benchmark
with tab_bench:
    if not run:
        st.info("No results yet. Choose models in the sidebar and click **Run benchmark**.")
    else:
        st.subheader("Recommendations")
        for line in recommendation_sentences(run["results"]):
            st.success(line)
        if run.get("judge_error"):
            st.warning("Judge problem: " + run["judge_error"])

        lb = leaderboard(run)
        st.subheader("Leaderboard")
        for _, row in lb.iterrows():
            if row.get("Warning"):
                st.warning("%s: %s (%.1f GB of %.1f GB)" % (row["Model"], row["Warning"], row["Peak RAM (GB)"],
                                                            run["system"]["ram_total_gb"]))
        st.dataframe(lb, hide_index=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Speed: average tokens/sec (higher is better)")
            st.bar_chart(lb.set_index("Model")["Avg tokens/sec"])
        with col_b:
            st.caption("Overall pass rate (%)")
            st.bar_chart(lb.set_index("Model")["Pass rate (%)"])

        # Optional Gemma 4 explanation (only aggregated numbers are used)
        if st.button("Explain these results with Gemma 4"):
            with st.spinner("Asking %s..." % JUDGE_MODEL):
                st.session_state["explanation"] = gemma_explain(run)
        if st.session_state.get("explanation"):
            st.info(st.session_state["explanation"])

        df_all = pd.DataFrame(run["results"])
        st.subheader("Failed calls")
        failed = df_all[df_all["error"].notna()][["model", "test_id", "error"]]
        if failed.empty:
            st.write("No failed calls. 🎉")
        else:
            st.dataframe(failed, hide_index=True)

        with st.expander("All answers"):
            cols = ["model", "task_type", "test_id", "passed", "time_s", "tokens_per_sec", "judge_score", "answer"]
            st.dataframe(df_all[cols], hide_index=True)

        # ---- history comparison
        st.subheader("History comparison")
        if len(run_files) < 2:
            st.write("Save at least two runs to compare them.")
        else:
            names = [os.path.basename(f) for f in run_files]
            ca, cb = st.columns(2)
            first = ca.selectbox("Older run (A)", names, index=max(len(names) - 2, 0))
            second = cb.selectbox("Newer run (B)", names, index=len(names) - 1)
            st.dataframe(compare_runs(load_json(os.path.join(RESULTS_DIR, first)),
                                      load_json(os.path.join(RESULTS_DIR, second))), hide_index=True)
            st.caption("Average tokens/sec per model across all saved runs")
            st.line_chart(history_tokens_per_sec(run_files))

        # ---- exports
        st.subheader("Export")
        e1, e2, e3 = st.columns(3)
        e1.download_button("Export anonymous profile (JSON)", json.dumps(anonymous_profile(run), indent=2),
                           "pickmymodel_profile.json", "application/json")
        e2.download_button("Leaderboard (CSV)", lb.to_csv(index=False), "leaderboard.csv", "text/csv")
        e3.download_button("Leaderboard (Markdown)", df_to_markdown(lb), "leaderboard.md", "text/markdown")
        st.caption("The anonymous profile has CPU name, RAM, OS, model names and aggregated scores only. "
                   "No prompts, answers, file paths or username.")

# ================================================================ TAB 2: Ask
with tab_ask:
    st.subheader("Ask your own question")
    prompt = st.text_area("Paste your project problem or prompt", height=160, key="ask_prompt")
    if prompt.strip():
        st.info(hint_from_latest(prompt, RESULTS_DIR))   # instant hint, no model call

    mode = st.radio("Mode", ["Compare selected models", "Auto-answer with the best model"], horizontal=True)
    a1, a2, a3 = st.columns(3)
    ask_timeout = a1.number_input("Timeout (seconds)", 10, 900, 120, step=10, key="ask_timeout")
    ask_tokens = a2.number_input("Max tokens", 50, 2000, 400, step=50, key="ask_tokens")

    if mode == "Compare selected models":
        ask_models_sel = st.multiselect("Models to compare", installed, default=installed[:3], key="ask_models")
        ask_judge = a3.checkbox("Judge answers", value=True, key="ask_judge")
        reference = st.text_area("Optional: paste a reference answer from Claude or ChatGPT", height=100,
                                 help="The judge compares each local answer to it. It is used as data only.")
        go = st.button("Find the best model", type="primary", disabled=bool(ollama_error or not prompt.strip()))
        if go:
            if not ask_models_sel:
                st.warning("Select at least one model.")
            else:
                bar = st.progress(0.0, text="Starting...")
                result = ask_models(prompt, ask_models_sel, int(ask_timeout), int(ask_tokens), ask_judge,
                                    reference.strip() or None,
                                    lambda d, t, label: bar.progress(d / max(t, 1), text="[%d/%d] %s" % (d, t, label)))
                bar.empty()
                st.session_state["ask_result"] = ("compare", result)
    else:
        gemma_router = a3.checkbox("Use Gemma 4 to detect the task type", value=False,
                                   help="Smarter than keywords but loads Gemma 4 first, so it is slower.")
        go = st.button("Answer with the best model", type="primary", disabled=bool(ollama_error or not prompt.strip()))
        if go:
            with st.spinner("Choosing a model and answering..."):
                st.session_state["ask_result"] = ("auto", auto_answer(prompt, installed, RESULTS_DIR,
                                                                      int(ask_timeout), int(ask_tokens), gemma_router))

    # ---- show the stored result (survives reruns)
    stored = st.session_state.get("ask_result")
    if stored:
        kind, res = stored
        if kind == "compare":
            (st.success if res["winner"] else st.error)(res["verdict"])
            if res.get("judge_error"):
                st.warning("Judge problem: " + res["judge_error"])
            table = pd.DataFrame(res["rows"])[["model", "judge_score", "time_s", "tokens_per_sec", "error"]]
            st.dataframe(table, hide_index=True)
            for row in res["rows"]:
                with st.expander("Answer from " + row["model"]):
                    st.write(row["answer"] or "(no answer) " + str(row["error"]))
            if res.get("path"):
                st.caption("Saved to results/" + os.path.basename(res["path"]))
        else:
            if res["error"]:
                st.error("%s failed: %s" % (res["model"], res["error"]))
            else:
                st.success("Answered by %s. %s" % (res["model"], res["reason"]))
                st.caption("%.1f s, %.1f tokens/sec" % (res["time_s"], res["tokens_per_sec"]))
                st.write(res["answer"])
