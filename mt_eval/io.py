"""Output layer — byte-compatible with the jais3 lm-evaluation-harness fork.

Everything the downstream pipeline (judge_vllm.py, merge_results.py, eval_status.py,
the dashboard) consumes is produced here, in exactly the shape the fork produces it:

* ``samples_<task>_<date_id>.jsonl``  — one record per generation, schema mirrored from
  ``lm_eval/evaluator.py`` (~L628) + ``loggers/evaluation_tracker.py`` (arguments sanitization).
  Reference sample: /scratch/mostafa.awad/sample.jsonl.
* ``results_<date_id>.json``          — aggregated results file mirrored from a real run's
  output (same top-level keys, same ``config`` block shape).
* Path resolution mirrors ``EvaluationTracker``: an ``--output_path`` ending in ``.json``
  writes into its parent dir with that stem (the pipeline always passes ``<dir>/results.json``);
  otherwise into ``<output_path>/<model_name_sanitized>/``.

Hashes: sha256 hexdigest of the utf-8 string, identical inputs to the fork
(doc -> json.dumps(doc, indent=2, default=handle_non_serializable, ensure_ascii=False);
prompt -> the rendered arg_0 string; target -> str(target)).
"""

import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path


def hash_string(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def handle_non_serializable(o):
    # mirrors lm_eval.utils.handle_non_serializable (numpy optional here)
    try:
        import numpy as np

        if isinstance(o, (np.int64, np.int32)):
            return int(o)
    except ImportError:
        pass
    if isinstance(o, set):
        return list(o)
    return str(o)


def sanitize_model_name(model_name: str) -> str:
    # mirrors lm_eval.utils.sanitize_model_name
    return re.sub(r"[\"<>:/|\\?*\[\]]+", "__", model_name)


def make_sample_record(
    doc_id: int,
    doc: dict,
    target: str,
    prompt: str,
    gen_kwargs: dict,
    resp: str,
    metadata: dict,
    metric_name: str,
    filter_key: str = "none",
) -> dict:
    """One samples_*.jsonl record, keys in the fork's serialization order.

    ``resp`` is the post-processed generation (think-stripped for thinking runs) — it becomes
    both resps[0][0] and filtered_resps[0] (no extraction filter on judged multi-turn tasks;
    filter name "none" like any lm-eval task without a filter pipeline).
    ``gen_kwargs`` is the task-defaults-updated-by-CLI dict, recorded verbatim as arg_1
    (i.e. pre-modify_gen_kwargs, exactly what lm-eval records from Instance.args).
    The trailing ``<metric_name>: 0.0`` is the placeholder the judge overwrites at scoring
    time (judge_vllm writes acc,none into results_*.json; per-sample placeholder mirrors
    lm-eval's example.update(metrics)).
    """
    return {
        "doc_id": doc_id,
        "doc": doc,
        "target": target,
        "arguments": {"gen_args_0": {"arg_0": prompt, "arg_1": dict(gen_kwargs)}},
        "resps": [[resp]],
        "filtered_resps": [resp],
        "filter": filter_key,
        "metrics": [metric_name],
        "doc_hash": hash_string(
            json.dumps(doc, indent=2, default=handle_non_serializable, ensure_ascii=False)
        ),
        "prompt_hash": hash_string(prompt),
        "target_hash": hash_string(str(target)),
        "generation_metadata": [[metadata]],
        metric_name: 0.0,
    }


def compute_task_hash(samples: list) -> str:
    # mirrors EvaluationTracker.save_results_aggregated's cumulative per-task hash
    return hash_string(
        "".join(s["doc_hash"] + s["prompt_hash"] + s["target_hash"] for s in samples)
    )


def _pretty_env_info() -> str:
    try:
        from torch.utils.collect_env import get_pretty_env_info

        return get_pretty_env_info()
    except Exception:
        return ""


def _transformers_version():
    try:
        import transformers

        return transformers.__version__
    except Exception:
        return None


def _lm_eval_version():
    try:
        import lm_eval

        return getattr(lm_eval, "__version__", None)
    except Exception:
        return None


def build_results_dict(
    task_results: dict,
    task_configs: dict,
    task_versions: dict,
    higher_is_better: dict,
    n_samples: dict,
    config_block: dict,
    model_info: dict,
    task_hashes: dict,
    start_unix: float,
    total_seconds: float,
    system_instruction: str = None,
) -> dict:
    """The aggregated results_*.json, same top-level keys (and order) as a real fork run."""
    return {
        "results": task_results,
        "group_subtasks": {t: [] for t in task_results},
        "configs": task_configs,
        "versions": task_versions,
        "n-shot": {t: 0 for t in task_results},
        "higher_is_better": higher_is_better,
        "n-samples": n_samples,
        "config": config_block,
        "git_hash": None,
        "date": start_unix,
        "pretty_env_info": _pretty_env_info(),
        "transformers_version": _transformers_version(),
        "lm_eval_version": _lm_eval_version(),
        "upper_git_hash": None,
        "tokenizer_pad_token": model_info.get("tokenizer_pad_token"),
        "tokenizer_eos_token": model_info.get("tokenizer_eos_token"),
        "tokenizer_bos_token": model_info.get("tokenizer_bos_token"),
        "eot_token_id": model_info.get("eot_token_id"),
        "max_length": model_info.get("max_length"),
        "task_hashes": task_hashes,
        "model_source": "vllm",
        "model_name": model_info.get("model_name"),
        "model_name_sanitized": sanitize_model_name(model_info.get("model_name") or ""),
        "system_instruction": system_instruction,
        "system_instruction_sha": (
            hash_string(system_instruction) if system_instruction else None
        ),
        "fewshot_as_multiturn": False,
        "chat_template": model_info.get("chat_template"),
        "chat_template_sha": (
            hash_string(model_info["chat_template"])
            if model_info.get("chat_template")
            else None
        ),
        "total_evaluation_time_seconds": str(total_seconds),
        "groups": {},
    }


class OutputWriter:
    """Resolves --output_path and writes files exactly like EvaluationTracker.

    The pipeline invokes lm-eval with ``--output_path <ckpt_results_dir>/results.json``:
    suffix == .json -> results file is ``<dir>/results_<date_id>.json`` and samples go to
    ``<dir>/samples_<task>_<date_id>.jsonl``. date_id is minted once at save time
    (datetime.now().isoformat() with ':' -> '-'), shared by the results file and every
    samples file of the run — judge_vllm/eval_status key on that.
    """

    def __init__(self, output_path: str, model_name: str):
        path = Path(output_path)
        if path.suffix == ".json":
            self.dir = path.parent
            self.results_stem = path.stem
        else:
            self.dir = path / sanitize_model_name(model_name)
            self.results_stem = "results"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.date_id = None

    def write(self, results_dict: dict, samples_by_task: dict) -> Path:
        self.date_id = datetime.now().isoformat().replace(":", "-")
        results_file = self.dir / f"{self.results_stem}_{self.date_id}.json"
        results_file.write_text(
            json.dumps(
                results_dict, indent=2, default=handle_non_serializable, ensure_ascii=False
            ),
            encoding="utf-8",
        )
        for task, samples in samples_by_task.items():
            samples_file = self.dir / f"samples_{task}_{self.date_id}.jsonl"
            with open(samples_file, "a", encoding="utf-8") as fout:
                for sample in samples:
                    fout.write(
                        json.dumps(
                            sample, default=handle_non_serializable, ensure_ascii=False
                        )
                        + "\n"
                    )
        return results_file
