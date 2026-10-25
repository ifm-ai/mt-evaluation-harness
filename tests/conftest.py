"""Shared fixtures and markers for the mt_eval test suite.

Three tiers, so the fast tier runs anywhere:

* unmarked        — pure unit tests. No GPU, no cluster paths, no network.
* needs_benchmarks — reads the real benchmark files under LOCAL_BENCHMARK_PATH.
* needs_pipeline   — imports the parent jais-3-evals repo (configs, judge_registry).
* needs_checkpoints — loads a real tokenizer from /jais3 (still CPU-only, no vLLM).

Marked tests skip themselves when what they need is absent, so a plain `pytest` is green
off-cluster and the same command exercises everything on it.
"""

import json
import os
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parent.parent
PIPELINE = HARNESS.parent  # the jais-3-evals repo root, when we are a submodule of it
TESTDATA = Path(__file__).resolve().parent / "testdata"

# Import mt_eval from this checkout rather than whichever copy pip installed, so the tests
# validate the code you are editing. Mirrors the PYTHONPATH override documented in README.
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

BENCHMARK_PATH = Path(
    os.environ.get("LOCAL_BENCHMARK_PATH", "/jais3/data/jais_2/benchmarks")
)

# One checkpoint per template family: ifm/bbq, and xllm/sft.
FAMILY_CKPTS = [
    "/jais3/checkpoints/bbq-4b-mid3_v3-final",
    "/jais3/checkpoints/jais3_experiments/sft_eval/"
    "A8_A8x5c17191lbWikipedia_cb_sft_reasoningv9.1_globalShuffle_checkpoint_5521_to_hf",
]


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "needs_benchmarks: requires the benchmark data under LOCAL_BENCHMARK_PATH"
    )
    config.addinivalue_line(
        "markers", "needs_pipeline: requires the parent jais-3-evals repo on sys.path"
    )
    config.addinivalue_line(
        "markers", "needs_checkpoints: requires a real tokenizer from /jais3/checkpoints"
    )


def pytest_collection_modifyitems(config, items):
    """Skip a marked test when its prerequisite is missing, with the reason spelled out."""
    have_benchmarks = BENCHMARK_PATH.is_dir()
    have_pipeline = (PIPELINE / "configs.py").is_file()
    for item in items:
        if "needs_benchmarks" in item.keywords and not have_benchmarks:
            item.add_marker(pytest.mark.skip(reason=f"no benchmark data at {BENCHMARK_PATH}"))
        if "needs_pipeline" in item.keywords and not have_pipeline:
            item.add_marker(pytest.mark.skip(reason=f"no jais-3-evals repo at {PIPELINE}"))
        if "needs_checkpoints" in item.keywords:
            missing = [p for p in FAMILY_CKPTS if not Path(p).exists()]
            if missing:
                item.add_marker(
                    pytest.mark.skip(reason=f"checkpoint not present: {missing[0]}")
                )


@pytest.fixture(scope="session")
def reference_records():
    """The vendored lm-eval reference records — the output-faithfulness contract.

    Trimmed from a real fork run (originally /scratch/mostafa.awad/sample.jsonl) to one
    non-thinking and one thinking record, copied verbatim: key ORDER and the sha256 hashes
    are what we assert against, so this file must never be reformatted or re-sorted.
    """
    path = TESTDATA / "reference_sample.jsonl"
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    assert records, f"no records in {path}"
    return records


@pytest.fixture(scope="session")
def reference_non_thinking(reference_records):
    return next(
        r for r in reference_records if "reasoning" not in r["generation_metadata"][0][0]
    )


@pytest.fixture(scope="session")
def reference_thinking(reference_records):
    return next(r for r in reference_records if "reasoning" in r["generation_metadata"][0][0])


@pytest.fixture
def pipeline_on_path():
    """Put the parent repo on sys.path for the tests that assert its contracts."""
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    return PIPELINE
