"""The thinking/grading policy from README, asserted rule by rule.

Pure unit tests: no tokenizer, no benchmark data, no parent repo. These are the checks that
must stay green everywhere, because they encode what a thinking generation contributes to
history versus what it contributes to judging.
"""

import pytest

from mt_eval.model import THINKING_CLOSING_TAGS, Generation, postprocess_response_text
from mt_eval.thinking import ThinkingHistory, missing_close_trace, split_trace

LOW = {"reasoning_effort": "low"}
HIGH = {"reasoning_effort": "high"}


# ── response normalization: only an opened tag plus a real close gets split ──────────

@pytest.mark.parametrize(
    "raw, prompt, think_end_token, chat_template_args, expected, why",
    [
        (
            "private trace</ifm|think_faster>FINAL ANSWER",
            "<ifm|think_faster>\n",
            None,
            LOW,
            "FINAL ANSWER",
            "low BBQ with a valid trace: the scorer sees the final answer only",
        ),
        (
            "unfinished private trace",
            "<ifm|think_faster>\n",
            None,
            LOW,
            "unfinished private trace",
            "low BBQ missing its close: the raw trace stays available as the judging fallback",
        ),
        (
            "Explain </think> literally",
            "<think_faster>\n</think_faster>\n",
            None,
            LOW,
            "Explain </think> literally",
            "low SFT direct answer: a tag inside the answer text must not trigger a split",
        ),
        (
            "HIGH-EFFORT FINAL ANSWER",
            "<think>\n",
            THINKING_CLOSING_TAGS,
            HIGH,
            "HIGH-EFFORT FINAL ANSWER",
            "high effort, already split by the wrapper: unchanged",
        ),
        (
            "Explain </think> literally",
            "<assistant>\n",
            None,
            {},
            "Explain </think> literally",
            "non-thinking model: tag-like answer text is left alone",
        ),
    ],
)
def test_postprocess_response_text(
    raw, prompt, think_end_token, chat_template_args, expected, why
):
    assert (
        postprocess_response_text(raw, prompt, [], think_end_token, chat_template_args)
        == expected
    ), why


# ── rules 5 + 6: telling an unfinished trace apart from a direct low-effort answer ───

HIGH_UNFINISHED = Generation(
    text="",
    raw="unfinished high trace",
    prompt="<think>\n",
    metadata={
        "has_think_end_token": False,
        "matched_think_end_token": None,
        "reasoning": "unfinished high trace",
    },
)
LOW_UNFINISHED = Generation(
    text="unfinished low trace",
    raw="unfinished low trace",
    prompt="<ifm|think_faster>\n",
    metadata={},
)
LOW_SFT_DIRECT = Generation(
    text="DIRECT ANSWER",
    raw="DIRECT ANSWER",
    prompt="<think_faster>\n</think_faster>\n",
    metadata={},
)
CLOSED_TRACE = Generation(
    text="FINAL ANSWER",
    raw="trace</ifm|think_faster>FINAL ANSWER",
    prompt="<ifm|think_faster>\n",
    metadata={},
)


def test_high_missing_close_is_unfinished_reasoning():
    """Rule 6: metadata identifies the usual high-effort missing-close case."""
    assert (
        missing_close_trace(HIGH_UNFINISHED, THINKING_CLOSING_TAGS)
        == "unfinished high trace"
    )


def test_low_missing_close_is_detected_from_the_open_prompt():
    """Rule 6: low-effort BBQ has no reasoning metadata, so the prompt is the only signal."""
    assert missing_close_trace(LOW_UNFINISHED) == "unfinished low trace"


def test_low_sft_direct_answer_is_not_unfinished_reasoning():
    """Rule 5: the template closed thinking in the prompt, so a missing tag means nothing."""
    assert missing_close_trace(LOW_SFT_DIRECT) is None


def test_closed_trace_is_not_unfinished_reasoning():
    assert missing_close_trace(CLOSED_TRACE) is None


def test_split_trace_separates_a_closed_trace():
    """Rule 1: a recognized close splits the generation into trace + answer."""
    assert split_trace(CLOSED_TRACE) == ("trace", "FINAL ANSWER")


def test_split_trace_reports_no_trace_for_a_direct_answer():
    assert split_trace(LOW_SFT_DIRECT) == ("", "DIRECT ANSWER")


# ── rules 4 + 7: judging text and history content stay separate ──────────────────────

class StructuredHistoryModel:
    """The low-effort template surface ThinkingHistory touches, and nothing else."""

    chat_template_args = LOW
    think_end_token = None
    history_assistant_wrap = "<think_faster>\n</think_faster>\n{content}"

    def apply_chat_template(self, messages):
        return "\n".join(
            str(message.get("think_faster", "")) + message.get("content", "")
            for message in messages
        )

    def history_content(self, generation):
        return generation.text


@pytest.fixture
def structured_model():
    return StructuredHistoryModel()


def test_unfinished_trace_goes_to_the_thinking_field_not_visible_content(structured_model):
    """Rule 4 + 7: the trace is the judge input, but history content must be empty."""
    with ThinkingHistory(structured_model) as history:
        message = history.from_generation(LOW_UNFINISHED)
    assert message == {
        "role": "assistant",
        "content": "",
        "think_faster": "unfinished low trace",
    }


def test_direct_answer_stays_visible_content_with_an_empty_field(structured_model):
    """Rule 5: a direct low-effort answer is an answer, with no trace to carry."""
    with ThinkingHistory(structured_model) as history:
        message = history.from_generation(LOW_SFT_DIRECT)
    assert message == {
        "role": "assistant",
        "content": "DIRECT ANSWER",
        "think_faster": "",
    }


def test_scripted_turn_gets_an_empty_thinking_field(structured_model):
    """A dataset-scripted assistant turn has no trace, so the field renders empty."""
    with ThinkingHistory(structured_model) as history:
        message = history.message({"role": "assistant", "content": "scripted a1"})
    assert message == {"role": "assistant", "content": "scripted a1", "think_faster": ""}


def test_user_turns_pass_through_untouched(structured_model):
    with ThinkingHistory(structured_model) as history:
        message = history.message({"role": "user", "content": "q"})
    assert message == {"role": "user", "content": "q"}


def test_context_manager_restores_the_empty_tag_wrap(structured_model):
    """The structured field and the empty-tag wrap are alternatives, never both — and a
    model instance shared across tasks must come out unchanged."""
    original = structured_model.history_assistant_wrap
    with ThinkingHistory(structured_model) as history:
        assert history.key == "think_faster"
        assert structured_model.history_assistant_wrap == "{content}"
    assert structured_model.history_assistant_wrap == original
