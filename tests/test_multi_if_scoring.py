"""Multi-IF scoring faithfulness, pinned to the pre-refactor standalone scorer.

Multi-IF used to be scored by a separate `score_multi_if.py` run after generation; the metric
is now computed in-task (`mt_eval/tasks/multi_if/utils.py`). These tests pin the arithmetic to
what that scorer produced, so the aggregation can never drift silently.

The IFEval checkers themselves are **nondeterministic** — `lm_eval/tasks/ifeval/instructions.py`
calls `langdetect.detect()`, which samples, so ~0.3% of records flip between identical runs and
the headline accuracy moves in the 4th decimal. That is a property of the checkers, not of this
code, and it makes a live-vs-live comparison useless for detecting a port bug. So the fixture
carries the checker outputs *recorded once*, and the tests replay those: same inputs, same
arithmetic, exact equality.
"""

import json

import pytest

from mt_eval.tasks.multi_if.utils import (
    METRIC_NAMES,
    per_sample_score,
    results_entry,
    score_records,
)

from .conftest import TESTDATA


@pytest.fixture(scope="module")
def golden():
    return json.loads((TESTDATA / "multi_if_golden.json").read_text(encoding="utf-8"))


@pytest.fixture
def replay(golden):
    """(records, process_results, agg_inst_level_acc) replaying the recorded checker outputs."""
    records = json.loads(json.dumps(golden["records"]))  # fresh copy; score_records annotates
    outputs = golden["checker_outputs"]
    state = {"i": 0}

    def process_results(doc, responses):
        out = outputs[state["i"]]
        state["i"] += 1
        return out

    def agg_inst_level_acc(items):
        """lm-eval's aggregation: flatten the per-record instruction lists, then mean."""
        flat = [x for item in items for x in item]
        return sum(flat) / len(flat) if flat else 0.0

    return records, process_results, agg_inst_level_acc


@pytest.fixture
def scored(replay):
    records, process_results, agg = replay
    metrics, diagnostics = score_records(records, process_results, agg, annotate=True)
    return records, metrics, diagnostics


def test_results_entry_matches_the_old_scorer_exactly(scored, golden):
    """Every key and value the standalone scorer wrote, reproduced bit for bit."""
    records, metrics, diagnostics = scored
    # round-trip through JSON, which is how the entry reaches results_*.json — it also makes
    # the diagnostics' per-turn keys strings on both sides (JSON has no integer keys)
    entry = json.loads(json.dumps(results_entry("multi_if", metrics, diagnostics, len(records))))
    expected = golden["expected_entry"]

    # acc_stderr,none is the one intentional change: the old scorer always wrote 0.0
    assert set(entry) == set(expected)
    for key in sorted(set(expected) - {"acc_stderr,none"}):
        assert entry[key] == expected[key], key


def test_headline_accuracy_is_the_mean_over_turns(scored, golden):
    _, metrics, _ = scored
    assert metrics["overall"]["multi_if_accuracy"] == golden["expected_entry"]["acc,none"]


def test_per_turn_metrics_match(scored, golden):
    _, metrics, _ = scored
    for turn, four in metrics["per_turn"].items():
        for name in METRIC_NAMES:
            assert four[name] == golden["expected_entry"][f"{name}_turn{turn},none"], (turn, name)


def test_diagnostics_match(scored, golden):
    """Empty / length-truncated / missing-close counts per turn — the truncation early warning."""
    _, _, diagnostics = scored
    assert {str(k): v for k, v in diagnostics.items()} == golden["expected_diagnostics"]


def test_per_sample_annotation_matches(scored, golden):
    """Each record carries its own breakdown and its real metric value, not the placeholder."""
    records, _, _ = scored
    for record, expected in zip(records, golden["expected_annotated"]):
        assert record["multi_if_score"] == expected["multi_if_score"]
        assert record[record["metrics"][0]] == expected["metric_value"]


def test_annotate_false_leaves_records_untouched(replay):
    records, process_results, agg = replay
    score_records(records, process_results, agg, annotate=False)
    assert all("multi_if_score" not in record for record in records)
    assert all(record["multi_if_accuracy"] == 0.0 for record in records)


def test_stderr_is_reported_for_the_headline(scored):
    """The one deliberate improvement over the old scorer, which hardcoded 0.0."""
    records, metrics, diagnostics = scored
    entry = results_entry("multi_if", metrics, diagnostics, len(records))
    assert entry["acc_stderr,none"] > 0.0
    assert metrics["stderr"] == entry["acc_stderr,none"]


# ── the per-record roll-up, independent of the fixture ───────────────────────────────

def test_per_sample_score_is_the_mean_of_four():
    out = {
        "prompt_level_strict_acc": True,
        "prompt_level_loose_acc": False,
        "inst_level_strict_acc": [True, True, False, False],
        "inst_level_loose_acc": [True, False, False, False],
    }
    score = per_sample_score(out)
    assert score["inst_level_strict_acc_mean"] == 0.5
    assert score["inst_level_loose_acc_mean"] == 0.25
    # mean of (1, 0, 0.5, 0.25)
    assert score["multi_if_accuracy"] == pytest.approx(0.4375)


def test_per_sample_score_keeps_raw_instruction_lists():
    out = {
        "prompt_level_strict_acc": False,
        "prompt_level_loose_acc": False,
        "inst_level_strict_acc": [True, False],
        "inst_level_loose_acc": [False, False],
    }
    score = per_sample_score(out)
    assert score["inst_level_strict_acc"] == [True, False]
    assert score["inst_level_loose_acc"] == [False, False]
