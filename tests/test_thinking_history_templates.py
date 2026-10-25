"""Multi-turn history rendering against the real chat templates, one per family.

Tokenizer only — no vLLM, no GPU — but it needs the actual checkpoints, because the whole
point is that the templates disagree and only the template can tell us what it accepts.

This covers the failure that motivated mt_eval/thinking.py: ifm/bbq templates raise
"Assistant message is missing a thinking field" for ANY content-only assistant message, which
made multi-turn impossible there until history started carrying the structured field.
"""

import pytest

from mt_eval.model import _HISTORY_WRAP_BY_EFFORT, postprocess_generated_text
from mt_eval.thinking import CLOSING_TAGS, ThinkingHistory, split_trace

from .conftest import FAMILY_CKPTS

pytestmark = pytest.mark.needs_checkpoints

# effort=None is the real gemma-style config: the pipeline sets no reasoning_effort, so the
# template takes no thinking field and history must stay on the empty-tag path.
EFFORTS = ["high", "low", None]
# think_end_token as the pipeline sets it: only for "think" runs, never for "instruct" runs --
# even though bbq/sft instruct runs still think at reasoning_effort=low.
THINK_END_TOKENS = [CLOSING_TAGS, None]


class StubGen:
    """Stands in for model.Generation."""

    def __init__(self, raw, text, metadata):
        self.raw, self.text, self.metadata = raw, text, metadata


class StubModel:
    """The parts of MultiTurnLM that history rendering touches."""

    def __init__(self, tokenizer, effort, think_end_token):
        self.tokenizer = tokenizer
        self.chat_template_args = {"reasoning_effort": effort} if effort else {}
        self.enable_thinking = bool(effort)
        self.think_end_token = think_end_token
        self.history_assistant_wrap = _HISTORY_WRAP_BY_EFFORT.get(effort, "{content}")

    def _wrap_assistant(self, content):
        wrap = self.history_assistant_wrap
        if wrap == "{content}":
            return content
        if (content or "").lstrip().startswith(("<think", "<|think", "<channel", "<ifm")):
            return content
        return wrap.format(content=content)

    def apply_chat_template(self, messages, add_generation_prompt=True):
        wrapped = [
            {**m, "content": self._wrap_assistant(m["content"])} if m["role"] == "assistant" else m
            for m in messages
        ]
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=self.enable_thinking,
            **self.chat_template_args,
        )
        try:
            return self.tokenizer.apply_chat_template(wrapped, **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self.tokenizer.apply_chat_template(wrapped, **kwargs)

    def history_content(self, generation):
        return (generation.raw or generation.text or "").strip()


def closing_tag_for(path, effort):
    """The tag this family actually emits at this effort."""
    if "bbq" in path:
        return "</ifm|think_faster>" if effort == "low" else "</ifm|think>"
    return "</think_faster>" if effort == "low" else "</think>"


@pytest.fixture(scope="module")
def tokenizers():
    """Load each family's tokenizer once; they are the expensive part of this module."""
    from transformers import AutoTokenizer

    return {
        path: AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        for path in FAMILY_CKPTS
    }


@pytest.fixture
def rendered(request, tokenizers):
    """(model, generation, trace, answer) for one (checkpoint, effort, think_end_token) cell."""
    path, effort, think_end_token = request.param
    model = StubModel(tokenizers[path], effort, think_end_token)

    # shaped like a real completion: the template already opened the tag, so the raw text is
    # TRACE + closing tag + answer
    raw = f"TRACE_TEXT{closing_tag_for(path, effort)}FINAL_ANSWER"
    text = postprocess_generated_text(raw, [], think_end_token)
    generation = StubGen(raw, text, {"reasoning": "", "matched_think_end_token": None})
    return model, generation


CELLS = [
    pytest.param(
        (path, effort, tet),
        id=f"{path.split('/')[-1][:24]}-effort={effort}-tet={'set' if tet else 'none'}",
    )
    for path in FAMILY_CKPTS
    for effort in EFFORTS
    for tet in THINK_END_TOKENS
]


@pytest.mark.parametrize("rendered", CELLS, indirect=True)
def test_self_conditioned_turn_two_renders(rendered):
    """The mt_bench flow: feed our own answer back as history and render the next turn.

    A template that rejects the message shape raises here — which is exactly what ifm/bbq did
    before history carried the structured thinking field.
    """
    model, generation = rendered
    with ThinkingHistory(model) as history:
        prompt = model.apply_chat_template(
            [
                {"role": "user", "content": "q1"},
                history.from_generation(generation),
                {"role": "user", "content": "q2"},
            ]
        )
    assert "FINAL_ANSWER" in prompt


@pytest.mark.parametrize("rendered", CELLS, indirect=True)
def test_three_turn_generated_history_renders(rendered):
    """The multi_if flow: three turns where EVERY prior assistant turn is one of ours.

    Unlike mt_bench (two turns) this accumulates generated history, so a template that only
    tolerates the first round-trip still fails here. multi_if used to hand-roll this instead of
    using ThinkingHistory; it now shares the path, and this keeps it that way.
    """
    model, generation = rendered
    conversation = []
    with ThinkingHistory(model) as history:
        for turn in range(3):
            conversation.append({"role": "user", "content": f"q{turn + 1}"})
            prompt = model.apply_chat_template(conversation)
            assert prompt
            conversation.append(history.from_generation(generation))
    # three user turns and three assistant turns survived the round trips
    assert [m["role"] for m in conversation] == ["user", "assistant"] * 3


@pytest.mark.parametrize("rendered", CELLS, indirect=True)
def test_scripted_history_renders(rendered):
    """The multichallenge flow: dataset-scripted assistant turns, no trace of our own."""
    model, _ = rendered
    scripted = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "scripted a1"},
        {"role": "user", "content": "u2"},
    ]
    with ThinkingHistory(model) as history:
        prompt = model.apply_chat_template([history.message(m) for m in scripted])
    assert "scripted a1" in prompt


@pytest.mark.parametrize("rendered", CELLS, indirect=True)
def test_trace_and_answer_split_cleanly(rendered):
    """Whatever the family, a closed trace splits into exactly TRACE_TEXT and FINAL_ANSWER."""
    model, generation = rendered
    trace, answer = split_trace(generation, model.think_end_token)
    assert (trace, answer) == ("TRACE_TEXT", "FINAL_ANSWER")


@pytest.mark.parametrize("rendered", CELLS, indirect=True)
def test_wrap_is_restored_after_the_context(rendered):
    """A model instance is shared across tasks, so it must come out as it went in."""
    model, generation = rendered
    expected = _HISTORY_WRAP_BY_EFFORT.get(
        model.chat_template_args.get("reasoning_effort"), "{content}"
    )
    with ThinkingHistory(model) as history:
        history.from_generation(generation)
    assert model.history_assistant_wrap == expected
