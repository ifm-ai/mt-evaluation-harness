"""Output faithfulness — the hard requirement.

Everything downstream of generation (judge_vllm.py, merge_results.py, eval_status.py, the
dashboard) consumes mt_eval's files as if the lm-evaluation-harness fork had written them. So
a record we build from a reference record's own fields must come back byte-identical in key
order and hashes. The reference is vendored in tests/testdata (see conftest).
"""

import json
import re

import pytest

from mt_eval.io import make_sample_record
from mt_eval.model import THINKING_CLOSING_TAGS

# the 8 keys thinking mode adds on top of the 4 non-thinking ones; deliberately WITHOUT
# answer_before_post_processing, which newer fork code emits but the reference does not
NON_THINKING_METADATA = {
    "finish_reason",
    "stop_reason",
    "generated_token_count",
    "word_count",
}
THINKING_EXTRA_METADATA = {
    "has_think_end_token",
    "think_end_token",
    "matched_think_end_token",
    "reasoning",
    "reasoning_word_count",
    "answer_word_count",
    "reasoning_token_count",
    "answer_token_count",
}


def rebuild(reference):
    """Feed a reference record's own fields back through our writer."""
    return make_sample_record(
        doc_id=reference["doc_id"],
        doc=reference["doc"],
        target=reference["target"],
        prompt=reference["arguments"]["gen_args_0"]["arg_0"],
        gen_kwargs=reference["arguments"]["gen_args_0"]["arg_1"],
        resp=reference["resps"][0][0],
        metadata=dict(reference["generation_metadata"][0][0]),
        metric_name=reference["metrics"][0],
        filter_key=reference["filter"],
    )


@pytest.fixture(params=["non_thinking", "thinking"])
def reference(request, reference_non_thinking, reference_thinking):
    return {
        "non_thinking": reference_non_thinking,
        "thinking": reference_thinking,
    }[request.param]


def test_top_level_key_order_is_identical(reference):
    assert list(rebuild(reference).keys()) == list(reference.keys())


@pytest.mark.parametrize("hash_key", ["doc_hash", "prompt_hash", "target_hash"])
def test_hashes_are_identical(reference, hash_key):
    assert rebuild(reference)[hash_key] == reference[hash_key]


def test_metadata_key_order_is_identical(reference):
    assert (
        list(rebuild(reference)["generation_metadata"][0][0].keys())
        == list(reference["generation_metadata"][0][0].keys())
    )


def test_non_thinking_metadata_has_exactly_four_keys(reference_non_thinking):
    assert set(reference_non_thinking["generation_metadata"][0][0]) == NON_THINKING_METADATA


def test_thinking_metadata_adds_exactly_eight_keys(reference_thinking):
    metadata = set(reference_thinking["generation_metadata"][0][0])
    assert metadata - NON_THINKING_METADATA == THINKING_EXTRA_METADATA


# ── the thinking split reproduces the reference exactly ──────────────────────────────

@pytest.fixture
def rejoined(reference_thinking):
    """Reassemble the raw generation from the reference's own split halves."""
    metadata = reference_thinking["generation_metadata"][0][0]
    raw = (
        metadata["reasoning"]
        + metadata["matched_think_end_token"]
        + "\n\n"
        + reference_thinking["resps"][0][0]
    )
    matches = list(re.finditer(THINKING_CLOSING_TAGS, raw))
    assert matches, "the reference's own closing tag must match THINKING_CLOSING_TAGS"
    return raw, matches[-1], metadata


def test_split_matches_the_reference_closing_tag(rejoined):
    _, last_match, metadata = rejoined
    assert last_match.group(0) == metadata["matched_think_end_token"]


def test_split_reproduces_the_reference_reasoning(rejoined):
    raw, last_match, metadata = rejoined
    assert raw[: last_match.start()] == metadata["reasoning"]


def test_split_reproduces_the_reference_answer(rejoined, reference_thinking):
    raw, last_match, _ = rejoined
    assert raw[last_match.end():].lstrip() == reference_thinking["resps"][0][0]


# ── filename and results-file contracts the pipeline relies on ───────────────────────

@pytest.mark.parametrize("task", ["multichallenge", "mt_bench", "mt_bench_arabic", "multi_if"])
def test_samples_filename_round_trips_through_get_taskname(task):
    """judge_registry.utils.get_taskname strips the prefix and the trailing date_id."""
    filename = f"samples_{task}_2026-08-11T12-00-00.123456.jsonl"
    assert filename[len("samples_"):].rpartition("_")[0] == task


@pytest.mark.parametrize("task", ["multichallenge", "mt_bench", "mt_bench_arabic", "multi_if"])
def test_samples_filename_matches_the_resume_regex(task):
    """helpers.get_remaining_tasks decides what to skip with this pattern."""
    filename = f"samples_{task}_2026-08-11T12-00-00.123456.jsonl"
    assert re.match(rf"samples_{re.escape(task)}_\d+.*\.jsonl$", filename)


@pytest.mark.needs_pipeline
def test_get_taskname_agrees_with_the_pipeline(pipeline_on_path):
    """Assert against the real implementation, not our restatement of it."""
    from judge_registry.utils import get_taskname

    assert get_taskname("samples_mt_bench_2026-08-11T12-00-00.123456.jsonl") == "mt_bench"
    assert get_taskname("samples_multi_if_2026-08-11T12-00-00.123456.jsonl") == "multi_if"


@pytest.mark.needs_pipeline
def test_reference_record_survives_the_judge_response_extraction(
    pipeline_on_path, reference_thinking
):
    """judge_registry.utils.get_model_response is what every judge prompt reads (policy
    rules 2-3): the answer when there is one, else the tail of the reasoning."""
    from judge_registry.utils import get_model_response

    assert get_model_response(reference_thinking) == reference_thinking["resps"][0][0].strip()

    empty = json.loads(json.dumps(reference_thinking))
    empty["resps"] = [[""]]
    fallback = get_model_response(empty)
    assert fallback, "an empty answer must fall back to the reasoning trace"
    assert empty["generation_metadata"][0][0]["reasoning"].strip().endswith(fallback[-40:])
