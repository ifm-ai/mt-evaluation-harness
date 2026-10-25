"""Task registry + base class.

A task is pure orchestration: it loads its docs, decides the message lists to generate
over (scripted history, self-conditioned turns, ...), and turns the model's Generations
into lm-eval-shaped sample records via io.make_sample_record. All model-aware behavior
(chat template, thinking config, history wrap, stop handling, metadata) lives in
MultiTurnLM. Adding a benchmark = one module with a @register class.

Naming contract: Task.NAME becomes the samples filename (samples_<NAME>_<date_id>.jsonl)
and the results key — judge_vllm derives the taskname back from the filename and routes it
through judge_registry.JUDGE_REGISTRY (matched by the entry's `benchmark` pattern), so NAME must
match what that expects (e.g. "multichallenge").
"""

import copy
import json
import os
from pathlib import Path

from ..io import make_sample_record

# configs.LOCAL_BENCHMARK_PATH in the parent pipeline
LOCAL_BENCHMARK_PATH = Path(
    os.environ.get("LOCAL_BENCHMARK_PATH", "/jais3/data/jais_2/benchmarks")
)

# stop strings shared by the pipeline's generate_until task configs (multichallenge.yaml);
# the model's decoded eos is appended at sampling time by the model wrapper
DEFAULT_UNTIL = [
    "<end_of_turn>",
    "<eos>",
    "</s>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|endoftext|>",
    "<|im_end|>",
]

TASK_REGISTRY = {}


def register(cls):
    TASK_REGISTRY[cls.NAME] = cls
    return cls


def get_task(name: str, **kwargs):
    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task {name!r}. Available: {sorted(TASK_REGISTRY)}"
        )
    return TASK_REGISTRY[name](**kwargs)


def read_jsonl(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as fin:
        return [json.loads(line) for line in fin if line.strip()]


class Task:
    NAME = None
    VERSION = 1.0
    # placeholder metric emitted per sample and in results (the judge overwrites the real
    # score into results_*.json as acc,none at judging time)
    METRIC = None
    UNTIL = DEFAULT_UNTIL

    def __init__(self, benchmark_path: Path = None, limit: int = None):
        self.benchmark_path = Path(benchmark_path or LOCAL_BENCHMARK_PATH)
        self.limit = limit

    # ------------------------------------------------------------------ interface

    def load_docs(self) -> list:
        raise NotImplementedError

    def run(self, model, cli_gen_kwargs: dict) -> list:
        """Generate everything for this task; return the list of sample records."""
        raise NotImplementedError

    # ------------------------------------------------------------------ shared helpers

    def merged_gen_kwargs(self, cli_gen_kwargs: dict) -> dict:
        """Task generation defaults updated by the CLI --gen_kwargs — the same merge (and
        resulting key order) lm-eval performs on config.generation_kwargs, recorded
        verbatim as arguments.gen_args_0.arg_1."""
        merged = {"until": list(self.UNTIL), "do_sample": False}
        merged.update(cli_gen_kwargs)
        return merged

    def record(self, doc_id, doc, target, generation, gen_kwargs):
        return make_sample_record(
            doc_id=doc_id,
            doc=doc,
            target=target,
            prompt=generation.prompt,
            gen_kwargs=copy.deepcopy(gen_kwargs),
            resp=generation.text,
            metadata=generation.metadata,
            metric_name=self.METRIC,
        )

    # ------------------------------------------------------------------ results entries

    def results_entry(self, n_samples: int) -> dict:
        """The pre-judge results_*.json entry for this task (judge_vllm adds acc,none)."""
        return {
            "name": self.NAME,
            "alias": self.NAME,
            "sample_len": n_samples,
            f"{self.METRIC},none": 0.0,
            f"{self.METRIC}_stderr,none": 0.0,
        }

    def higher_is_better(self) -> dict:
        return {self.METRIC: True}

    def task_config(self) -> dict:
        """The configs[<task>] block of results_*.json (informational; mirrors the shape
        of an lm-eval task config)."""
        return {
            "task": self.NAME,
            "output_type": "generate_until",
            "generation_kwargs": {"until": list(self.UNTIL), "do_sample": False},
            "metadata": {"version": self.VERSION},
        }


from . import multichallenge  # noqa: E402,F401  (populates TASK_REGISTRY)
from . import mt_bench  # noqa: E402,F401
from . import multi_if  # noqa: E402,F401
from . import aragen  # noqa: E402,F401
