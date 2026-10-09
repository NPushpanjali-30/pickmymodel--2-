---
name: pickmymodel
description: Benchmarks the open-weight models installed in Ollama on this machine and recommends the best local model for coding, summarizing, extraction or tool-calling. Use when the user asks which local or open-weight model to run, wants to compare Ollama models, or needs a model recommendation for a laptop with limited RAM.
license: MIT
compatibility: Requires Python 3.10+, a running Ollama server with at least one model pulled, and the PickMyModel repository dependencies (pip install -r requirements.txt).
metadata:
  author: pickmymodel
  version: "1.0"
---

# PickMyModel

Recommends the best locally installed Ollama model per task, based on real measurements
(speed, pass rate, load time, peak RAM) on this machine. Everything runs locally.

## Steps

1. Check for an existing result first (fast, no model calls):
   `python benchmark.py --latest-json`
   If it prints recommendations that are recent enough for the user, use them and stop.
2. If it prints an error saying there are no saved runs, or the user wants fresh numbers, run a benchmark.
   Start small on low-RAM machines:
   `python benchmark.py --tasks coding extraction --repeats 1`
   Add `--judge` only if the user wants an AI quality score (slower, uses `gemma4:e4b`).
3. Run `python benchmark.py --latest-json` again and read the `sentences` field.
4. Tell the user the recommended model per task in plain language, and mention any
   "WARNING: peak RAM" notes from the benchmark output.

## Notes

- Models run one at a time and are unloaded after each, so memory use stays low.
- If the output says Ollama is not reachable, ask the user to start Ollama (`ollama serve`).
- Run all commands from the repository root (the folder containing `benchmark.py`).
- Do not paste or share prompts or answers from `results/`; use "Export anonymous profile" in the dashboard for sharing.
