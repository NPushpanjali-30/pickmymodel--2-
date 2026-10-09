# PickMyModel

**Benchmark the open-weight models installed in Ollama on your own laptop and find the best one for each task.**
It runs on a 16 GB RAM, CPU-only Windows laptop, needs no paid API, and works fully offline.

## What it does

- Detects your installed Ollama models, your RAM and your CPU.
- Runs small, deterministic tests (coding, summarizing, extraction, tool-calling) on each model, **one model at a time**, and unloads each model afterwards.
- Measures speed (tokens/sec), response time, **model load time** and **peak RAM**, and warns when a model uses more than 85% of your RAM.
- Recommends the best model per task, for example:
  `Best for coding on this machine: qwen3:4b (100% pass rate, 19.3 tokens/sec)`
- Optional **LLM judge** (Gemma 4) that scores answers 1-5.
- **Ask tab:** paste your own project prompt, compare models side by side, or let the **auto-router** pick the best model for you.
- Compares saved runs over time, and exports an anonymous, shareable profile.

## Why it exists

"Which local model should I run?" depends on *your* hardware. Leaderboards test big GPUs, not a 16 GB laptop CPU.
PickMyModel answers the question with measurements from your own machine.

## How it differs from cloud assistants

| | Cloud assistants | PickMyModel |
|---|---|---|
| Privacy | Prompts leave your machine | **Private**: nothing leaves your laptop |
| Cost | Subscriptions or per-token fees | **Free**: open-weight models, no API keys |
| Connection | Needs internet | **Offline** after models are downloaded |
| Hardware | Hidden | **Hardware-aware**: speed, load time and RAM on your machine |
| Choice | One model | **Side-by-side** comparison of many models |

You can still paste a reference answer from Claude or ChatGPT into the Ask tab and see how close each local model gets.

## Install and run (Windows PowerShell)

### 1. Install Ollama and pull models

Install Ollama from <https://ollama.com/download>, then in PowerShell:

```powershell
ollama pull gemma4:e4b      # also used as the judge
ollama pull qwen3:4b
ollama pull llama3.2:3b
```

You can also pull GGUF models straight from Hugging Face, no extra code needed:

```powershell
ollama pull hf.co/<user>/<repo>
```

Model names change over time. Check <https://ollama.com/library> if a pull fails, and use any small models that fit your RAM.

### 2. Set up the project

```powershell
cd pickmymodel
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once and try again:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`

### 3. Run the benchmark (command line)

```powershell
python benchmark.py
python benchmark.py --models qwen3:4b llama3.2:3b --tasks coding extraction
python benchmark.py --judge --repeats 3 --timeout 90 --max-tokens 200
python benchmark.py --explain          # Gemma 4 explains the results in plain English
python benchmark.py --latest-json      # print recommendations from the latest saved run
```

| Flag | Meaning |
|---|---|
| `--models` | Models to test (default: all installed) |
| `--tasks` | Task types: `coding`, `summarizing`, `extraction`, `tool_calling` (default: all) |
| `--judge` | Also score answers 1-5 with `gemma4:e4b` (second phase) |
| `--timeout` | Seconds per call (default 60) |
| `--max-tokens` | Override the max output tokens |
| `--repeats` | Runs per test, averaged (default 1). A test passes if most repeats pass |

It prints system info, progress as `[done/total]`, the leaderboard, the recommendation sentences and the path of the saved file in `results/`.

### 4. Run the dashboard

```powershell
streamlit run app.py
```

The sidebar picks models, task types, judge, timeout and repeats, and starts the run. Use "Load older results" to open a saved run.

## The Ask tab

1. Paste your own project problem or prompt. An instant hint appears from your latest saved benchmark (no model call).
2. **Compare selected models:** each model answers one at a time. With the judge on (default here), the winner is the fastest model scoring 4 or more; otherwise the highest score. Without judge scores it names the fastest model.
3. **Auto-answer with the best model:** uses the latest benchmark recommendation for the task type (or the fastest installed model), answers, and tells you which model it used and why. Optionally let Gemma 4 detect the task type.
4. Optional **cloud baseline:** paste a reference answer from Claude or ChatGPT. The judge compares each answer to it twice, with the order swapped, and averages the scores. The pasted text is treated as data only.
5. Results are saved to `results/compare_<timestamp>.json`.

## Models used

- Any open-weight model you have installed in Ollama (embedding models are skipped).
- **Gemma 4 (`gemma4:e4b`)** is the optional judge, the optional task-type router and the results explainer. It always runs alone, after the other models have been unloaded.

## Add your own tests

Drop a `.json` file in `tests/`. It is a list of tests:

```json
[
  {
    "id": "my_test_1",
    "task_type": "coding",
    "prompt": "Write a Python function double(n) that returns n * 2. Reply with code only.",
    "check": "contains_keywords",
    "expected": ["def double", "return"],
    "min_matches": 2,
    "max_tokens": 128
  }
]
```

| Field | Meaning |
|---|---|
| `id` | Unique name |
| `task_type` | Any label. Built-in: `coding`, `summarizing`, `extraction`, `tool_calling` |
| `prompt` | What the model is asked |
| `check` | `json_valid`, `contains_keywords`, `exact_match` or `tool_call` |
| `expected` | Depends on the check (below) |
| `max_tokens` | Optional output limit (default 256) |

Checks:

- `json_valid`: `expected` is a list of required keys (may be empty). Markdown code fences and extra prose around the JSON are tolerated.
- `contains_keywords`: `expected` is a list of words, matched case-insensitively. Optional `min_matches` (default: all).
- `exact_match`: `expected` is a string. Case, quotes and a trailing `.` or `!` are ignored.
- `tool_call`: `expected` is `{"tool": "get_weather", "arguments": ["city"]}`. The answer must be JSON naming that tool with non-empty values for those argument names.

`<think>...</think>` blocks are removed before scoring.

## Agent Skill

`skills/pickmymodel/SKILL.md` follows the [Agent Skills](https://agentskills.io) format, so a coding agent can run the benchmark and read back the recommendation (`python benchmark.py --latest-json`).

## Share your results

In the dashboard, "Export anonymous profile" downloads a JSON file with CPU name, RAM size, OS, model names and aggregated scores. It contains no prompts, no answers, no file paths and no username. CSV and Markdown downloads of the leaderboard are also available.

## Tests

```powershell
pip install pytest
python -m pytest unit_tests
```

These use a mocked Ollama client, so they run without Ollama.

## Limitations

- The checks are simple (keywords, exact match, valid JSON) and the test set is small, so treat results as a guide, not a final verdict.
- The judge is a small model and may be biased. Averaging two orderings reduces position bias but does not remove it.
- Results depend on your hardware, background apps and Windows power mode. Plug in and use the "Best performance" mode for fair numbers.
- Peak RAM counts the memory of Ollama processes only.
- If a model cannot load during warm-up, its tests are recorded as failed and the run moves on.

## License

MIT. See [LICENSE](LICENSE).
