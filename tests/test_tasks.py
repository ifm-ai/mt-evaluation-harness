"""End-to-end task runs over the real benchmark files, with a mock model instead of vLLM.

This is the check that the multi-turn flows actually hold: MT-Bench conditioning turn 2 on the
model's own turn-1 answer, MultiChallenge generating only the final turn, and the writer
laying files out the way EvaluationTracker does.
"""

import json

import pytest

from mt_eval.io import OutputWriter
from mt_eval.model import Generation
from mt_eval.tasks import get_task

pytestmark = pytest.mark.needs_benchmarks

LIMIT = 3
CLI_GEN_KWARGS = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 20,
    "do_sample": True,
    "max_gen_toks": 8192,
}


class MockModel:
    """Stands in for MultiTurnLM: fixed text per generation, non-thinking metadata.

    A non-thinking model on purpose — ThinkingHistory's template probe finds no thinking
    field and leaves history on the plain-content path, which is the simplest contract to
    assert task flow against.
    """

    chat_template_args = {}
    think_end_token = None
    history_assistant_wrap = "{content}"

    def history_content(self, generation):
        return generation.text

    def generate_batch(self, conversations, gen_kwargs):
        generations = []
        for i, conversation in enumerate(conversations):
            prompt = (
                "\n".join(f"<{m['role']}>{m['content']}" for m in conversation)
                + "\n<assistant>"
            )
            generations.append(
                Generation(
                    text=f"mock answer {i}",
                    raw=f"mock answer {i}",
                    prompt=prompt,
                    metadata={
                        "finish_reason": "stop",
                        "stop_reason": "<|im_end|>",
                        "generated_token_count": 3,
                        "word_count": 3,
                    },
                )
            )
        return generations


@pytest.fixture(scope="module")
def mock_run():
    """Run every multi-turn task once, at --limit 3, and reuse the records."""
    model = MockModel()
    samples_by_task = {}
    for name in ("multichallenge", "mt_bench", "mt_bench_arabic"):
        samples_by_task[name] = get_task(name, limit=LIMIT).run(model, CLI_GEN_KWARGS)
    return samples_by_task


def test_multichallenge_emits_one_record_per_doc(mock_run):
    assert len(mock_run["multichallenge"]) == LIMIT


def test_mtbench_emits_two_records_per_question(mock_run):
    assert len(mock_run["mt_bench"]) == 2 * LIMIT


def test_mtbench_doc_ids_are_sequential(mock_run):
    records = mock_run["mt_bench"]
    assert [r["doc_id"] for r in records] == list(range(2 * LIMIT))


def test_mtbench_turn_two_is_self_conditioned(mock_run):
    """Turn 2's prompt must contain the model's own turn-1 answer, and prev_answer must be it."""
    first, second = mock_run["mt_bench"][0], mock_run["mt_bench"][1]
    assert "mock answer" in second["arguments"]["gen_args_0"]["arg_0"]
    assert second["doc"]["prev_answer"] == first["resps"][0][0]


def test_mtbench_reference_answers_are_carried_for_reference_categories(mock_run):
    """math/reasoning/coding rows need the gpt-4 reference for the judge's math rubric."""
    records = mock_run["mt_bench"]
    need_reference = ("math", "reasoning", "coding", "arena-hard-200")
    for record in records:
        if record["doc"]["category"] in need_reference:
            assert record["doc"]["reference"], record["doc"]["question_id"]


def test_gen_kwargs_merge_order_matches_lm_eval(mock_run):
    """arg_1 = task defaults updated by the CLI kwargs — same keys in the same order as a
    real fork record, since the merge order is part of the sample schema."""
    arg_1 = mock_run["mt_bench"][0]["arguments"]["gen_args_0"]["arg_1"]
    assert list(arg_1.keys()) == [
        "until",
        "do_sample",
        "temperature",
        "top_p",
        "top_k",
        "max_gen_toks",
    ]


def test_writer_resolves_output_path_like_the_tracker(mock_run, tmp_path):
    """`--output_path <dir>/results.json` becomes <dir>/results_<date_id>.json plus one
    samples file per task, exactly as EvaluationTracker lays it out."""
    writer = OutputWriter(str(tmp_path / "results.json"), "mock/model")
    results = {"results": {task: {"alias": task} for task in mock_run}}
    results_file = writer.write(results, mock_run)

    assert results_file.parent == tmp_path
    assert results_file.name.startswith("results_")

    samples_files = sorted(tmp_path.glob("samples_*.jsonl"))
    assert len(samples_files) == len(mock_run)

    loaded = json.loads(results_file.read_text(encoding="utf-8"))
    assert set(loaded["results"]) == set(mock_run)


@pytest.mark.needs_pipeline
def test_multichallenge_judge_prompt_builds_from_our_records(
    mock_run, tmp_path, pipeline_on_path
):
    """The judge must accept our records untouched: resolve the registry entry by task name
    and build a prompt that embeds both the response and the pass criteria."""
    from judge_registry.utils import get_task_entry

    entry = get_task_entry("multichallenge")
    assert entry and entry["benchmark"] == "multichallenge"

    build_prompt = entry["prompt_builder"]
    for record in mock_run["multichallenge"]:
        prompt = build_prompt(record)
        assert record["resps"][0][0] in prompt
        assert record["doc"]["TARGET_QUESTION"] in prompt
