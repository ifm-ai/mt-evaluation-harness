# mt-evaluation-harness (`mt_eval`)

Generation engine for multi-turn benchmarks, alongside `lm-evaluation-harness`. Generation stage
only: the CLI mirrors lm-eval's flags and the output files are byte-compatible with the fork, so
`judge_vllm.py`, `merge_results.py`, `eval_status.py` and the dashboard read them unchanged.
`configs.TASK_TO_HARNESS` routes tasks here.

Judged tasks are scored by `judge_vllm.py` through `judge_registry`. Programmatic tasks score
themselves in-task and write the metric into `results_*.json`, as lm-eval's `ifeval` does.

## Install

Editable install into every venv that can run a generation job, the same set `lm_eval` uses.
Without it, jobs fail with `mt-eval: command not found`.

```bash
<venv>/bin/pip install -e <repo>/mt-evaluation-harness --no-deps
```

`--no-deps` matters: these venvs carry patched vLLM builds that dependency resolution can replace.
`pyproject.toml` declares no hard runtime dependencies and imports `transformers`/`vllm` lazily.

The editable finder pins one absolute path regardless of where you launch from, so install from
the shared checkout. To run another copy:

```bash
PYTHONPATH=<your-copy>/mt-evaluation-harness python -m mt_eval --model vllm …
```

## Usage

`eval.py --sbatch` builds the whole job wrap (env, venv, results dir, harness command); there is
no runner script here.

```bash
python eval.py --sbatch --think google/gemma-4-E4B-it \
    --tasks mt_bench mt_bench_arabic multichallenge \
    --results-path /jais3/eval/results_optimized
```

`--limit 3` smokes it. Mixed task lists are fine: `create_eval_cmd` emits one command per harness.
To drive the CLI directly:

```bash
mt-eval --model vllm --apply_chat_template --log_samples --trust_remote_code \
  --model_args '{"pretrained": "google/gemma-4-E4B-it", "dtype": "bfloat16", "seed": 42,
                 "gpu_memory_utilization": 0.9, "tensor_parallel_size": 1,
                 "max_model_len": 32768, "log_completion_metadata": true,
                 "enable_thinking": true,
                 "think_end_token": "</think>|</think_fast>|</think_faster>|<channel\\|>|</ifm\\|think>"}' \
  --gen_kwargs 'temperature=1.0,do_sample=True,top_p=0.95,top_k=64,max_gen_toks=32768' \
  --tasks mt_bench,multichallenge \
  --output_path <model_dir>/results.json
```

`--model_args` takes the JSON `helpers.get_model_args` produces, or lm-eval csv `k=v`.
Wrapper-consumed keys: `pretrained`, `max_model_len`, `enable_thinking`, `chat_template_args`
(`{"reasoning_effort": "low"|"high"}` for jais-3/xllm), `think_end_token`,
`log_completion_metadata`, `history_assistant_wrap`; everything else goes to `vllm.LLM`.
`helpers.get_ckpt_venv` is the source of truth for the venv. In `think_end_token`, note the
doubled backslashes: inside JSON a lone `\|` is an invalid escape.

Judge and merge as usual afterwards.

### Context budget

A turn's answer becomes the next turn's prompt, so `helpers.resolve_multiturn_budget` sizes the
window to `per_turn × configs.TASK_NUM_TURNS[task]` and caps `max_gen_toks` at `per_turn`. Both
halves are required: with `max_gen_toks` equal to the window, a runaway turn-1 trace fills it and
later turns get about 1 token. If the model's config length or `--max-model-length` caps the
window, `per_turn` is divided instead. Single-turn tasks are unaffected.

## Layout

```
mt_eval/
  __main__.py    lm-eval-compatible CLI (python -m mt_eval)
  model.py       MultiTurnLM: vLLM + tokenizer, chat template, thinking config, history wrap,
                 stop handling, completion metadata
  thinking.py    trace/answer split + thinking-field history shapes
  io.py          samples_*.jsonl / results_*.json schema + writers
  tasks/         registry + base Task; one directory per benchmark, importing it registers it
pyproject.toml   packaging + the mt-eval console script
tests/           pytest suite; tests/testdata holds the fixtures
```

## Thinking and grading policy

What a thinking generation contributes to history versus to judging. History is
`mt_eval/thinking.py`; grading is `judge_registry/utils.py` in the parent repo.

Templates disagree on how a prior assistant turn carries reasoning, so `resolve_history_key`
renders a sentinel through the template and checks whether it comes back: xllm/jais-3 accepts a
structured `think` / `think_fast` / `think_faster` field, ifm/bbq requires one, gemma-4 takes none.
The field must match the effort (`high → think`, `medium → think_fast`, `low → think_faster`),
since the generation prompt has already opened that effort's tag. The structured field and the
empty-tag `history_assistant_wrap` are alternatives, never both.

1. Closing tag present: split into trace + answer. Trace to the thinking field, answer to visible
   content.
2. *(parent)* Grade the answer only, not the trace or tags.
3. *(parent)* No answer extractable: judge the reasoning instead, truncated to its last
   `REASONING_TAIL_NUM_CHARS_TO_KEEP` characters. Judging only; the trace stays out of visible
   content.
4. An unclosed trace used as the judging fallback still goes in the thinking field for later
   turns: `{"role": "assistant", "content": "", "think": raw_reasoning}`.
5. Low-effort SFT closes thinking in the prompt, so a completion with no closing tag is a direct
   answer, not unfinished reasoning. `missing_close_trace` tells this from rule 6 by the metadata
   and the rendered prompt.
6. BBQ reasons at low and high effort. Closing tag: split, grade the answer. No closing tag: raw
   generation is unfinished reasoning, judged as the fallback and kept in the thinking field.
7. Judge input and history content are independent:

   | | judge input | history content | history thinking |
   |---|---|---|---|
   | completed response | final answer | final answer | completed trace |
   | unfinished reasoning | raw trace | *(empty)* | raw trace |

`mt_eval`'s closing-tag set is wider than `configs.THINKING_CLOSING_TAGS`, which misses
`</ifm|think_fast>` / `</ifm|think_faster>`; without the superset an ifm template at medium/low
effort leaks the whole trace into the answer.

Two fork differences, invisible downstream: thinking metadata is keyed on `think_end_token` alone
rather than `enable_thinking OR think_end_token`, avoiding the `reasoning_effort=low` artifact
where `resps[0][0]` is `""`; and context-window fitting caps `max_tokens` directly instead of
relying on lm-eval's no-op truncation plus vLLM's clamp.

In-task scoring reuses lm-eval's IFEval checkers, which call `langdetect.detect()`. That samples,
so scores are not bit-reproducible: ~0.3% of records flip between identical runs, moving the
headline in the 4th decimal. Rankings are unaffected. Same for `ifeval` / `ifeval_ar`.

## Tests

`pytest`, laid out like `lm-evaluation-harness/tests/`. Marked tiers skip themselves when what
they need is absent.

| selector | n | needs |
|---|---|---|
| *(unmarked)* | 48 | nothing |
| `-m needs_benchmarks` | 8 | benchmark files under `LOCAL_BENCHMARK_PATH` |
| `-m needs_pipeline` | 3 | the parent repo importable |
| `-m needs_checkpoints` | 59 | a real tokenizer from `/jais3/checkpoints` (CPU-only) |

`test_thinking_policy.py` covers the rules above; `test_output_faithfulness.py` the record schema
against a vendored fork sample; `test_tasks.py` the task flows over real benchmark data;
`test_multi_if_scoring.py` pins in-task IFEval arithmetic using recorded checker outputs;
`test_thinking_history_templates.py` renders history through each family's real template.

`conftest.py` owns markers, skip reasons and fixtures, and puts this checkout ahead of the
installed `mt_eval` on `sys.path`.

## Adding a task

1. Stage the data under `LOCAL_BENCHMARK_PATH` in its own subdirectory.
2. Create `mt_eval/tasks/<name>/` with `task.py` and an `__init__.py` holding
   `from . import task`.
3. In `task.py`: subclass `Task`, set `NAME` (becomes the samples filename and results key) and
   `METRIC`, implement `load_docs()` and `run(model, cli_gen_kwargs)`, decorate with `@register`.
   `run()` drives `model.generate_batch(...)` and returns `self.record(...)` rows.
4. Build history with `ThinkingHistory`, never by hand: `from_generation(gen)` for a turn you
   generated, `message(m)` for a scripted one. Call it before mutating the generation.
5. Import the package at the bottom of `tasks/__init__.py`.
6. In the parent `configs.py`: `TASK_TO_HARNESS`, `TASK_MAX_MODEL_LEN`, `EN_TASKS`/`AR_TASKS`, and
   `TASK_NUM_TURNS` if it chains turns.
7. Scoring: either a `judge_registry` entry matching `NAME` with `METRIC` left as a placeholder, or
   in-task scoring returning real numbers from `results_entry()`, checkers kept in the task
   directory.
8. Add its flow to `test_tasks.py`, and to `test_thinking_history_templates.py` if it introduces a
   new history shape.
9. Smoke with `--limit 2` and check `finish_reason` for truncation.

Variants of one benchmark share the directory and a base class, differing only in `NAME` and the
data file.
