"""How a completed assistant turn is put back into multi-turn history.

A thinking model is prompted with the opening thinking tag already emitted by the chat
template, so its raw completion is "TRACE<close>ANSWER" (or just "ANSWER" when the template
emits an empty tag pair up front). Feeding that back as a later turn's history needs the
trace and the answer in whatever shape the template expects, and the templates disagree:

* xllm / jais-3 ("sft", `<think>`, `<|im_start|>`): renders a bare `content` as-is, and also
  accepts a structured `think` / `think_fast` / `think_faster` message field, which it wraps
  in the matching tags itself.
* ifm / bbq (`<ifm|think>`, `<|ifm|im_start|>`): *requires* one of think / think_fast /
  think_faster / reasoning / reasoning_content to be present as a message KEY, and raises
  "Assistant message is missing a thinking field" otherwise. Its content-sniffing fallback is
  unreachable (it sits behind an elif that only fires when a key exists but is not a string),
  so NOTHING written into `content` can satisfy it -- not the answer, not the answer wrapped
  in empty tags, not the raw trace with its tags embedded. All three raise.
* gemma-4 and other non-thinking templates: take no thinking field at all.

So the only representation that works everywhere is the structured field, and the only safe
way to pick it is to ask the template. `resolve_history_key` renders a sentinel and checks
whether it comes back, which also tells apart "template mentions the field" from "template
actually renders it on an ordinary assistant turn". None means no thinking field, and the
caller stays on the model's empty-tag `history_assistant_wrap` exactly as before.

The field must match the effort: each of think / think_fast / think_faster renders under its
own tag and the generation prompt opens the effort's tag, so any other field puts history
under a tag the model is not currently reasoning in.
"""

import re

# reasoning_effort -> the message field whose tag matches the tag the generation prompt opens
THINK_KEY_BY_EFFORT = {"high": "think", "medium": "think_fast", "low": "think_faster"}

# Every closing tag we know, as a regex. Superset of configs.THINKING_CLOSING_TAGS, which
# matches `</ifm|think>` but NOT `</ifm|think_fast>` / `</ifm|think_faster>` -- so on an ifm
# template at medium/low effort nothing recognises the end of the trace and the whole trace
# leaks into the answer.
CLOSING_TAGS = (
    r"</think>|</think_fast>|</think_faster>|<channel\|>|</ifm\|think(?:_fast|_faster)?>"
)

_PROBE_SENTINEL = "__mt_eval_thinking_probe__"
OPEN_THINKING_PROMPT = re.compile(
    r"<(?:ifm\|)?think(?:_fast|_faster)?>\s*$"
)


def resolve_history_key(model):
    """The thinking field this model's template renders, or None if it takes none.

    Tries the effort's own field first so a thinking run can never land on a mismatched tag,
    then the other spellings for templates that name the field differently.
    """
    chat_template_args = getattr(model, "chat_template_args", {}) or {}
    preferred = THINK_KEY_BY_EFFORT.get(chat_template_args.get("reasoning_effort", "high"))
    candidates = [preferred] if preferred else []
    candidates += [
        key
        for key in ("reasoning_content", "reasoning", "think", "think_fast", "think_faster")
        if key != preferred
    ]

    probe = [
        {"role": "user", "content": "probe"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "follow up"},
    ]
    for key in candidates:
        candidate = [*probe]
        candidate[1] = {**candidate[1], key: _PROBE_SENTINEL}
        try:
            if _PROBE_SENTINEL in model.apply_chat_template(candidate):
                return key
        except Exception:
            continue
    return None


def split_trace(generation, think_end_token=None):
    """(trace, answer) for a completed generation.

    Prefers the wrapper's own metadata split. Falls back to splitting the raw text on the
    closing tag, which is what saves the runs where the pipeline never set think_end_token:
    it only sets it for ckpt_type "think", while the bbq/sft override also makes "instruct"
    runs think (at reasoning_effort=low), leaving those generations unsplit.

    A closing tag is what separates a real trace from a direct answer: no tag means the
    template already emitted an empty tag pair and the model replied straight out, so the
    whole completion is the answer and there is no trace to carry.
    """
    metadata = getattr(generation, "metadata", None) or {}
    trace = (metadata.get("reasoning") or "").strip()
    if trace and metadata.get("matched_think_end_token"):
        return trace, generation.text

    raw = (getattr(generation, "raw", None) or "").strip()
    matches = list(re.finditer(think_end_token or CLOSING_TAGS, raw))
    if matches:
        last = matches[-1]
        return raw[: last.start()].strip(), raw[last.end():].lstrip()

    # No closing tag anywhere: a direct answer. Prefer the post-processed text, but fall back
    # to metadata.reasoning, which is where the parser parks a completion it could not split.
    return "", (generation.text.strip() or trace or raw)


def missing_close_trace(generation, think_end_token=None):
    """Return an unfinished reasoning trace, or None for a direct answer.

    A no-close completion is ambiguous: low-effort SFT closes an empty thinking block in
    the prompt and generates a direct answer, while BBQ leaves thinking open and generates
    a trace.  Completion metadata identifies the usual high-effort missing-close case; the
    rendered prompt identifies low-effort BBQ runs whose metadata has no reasoning fields.
    """
    metadata = getattr(generation, "metadata", None) or {}
    raw = (getattr(generation, "raw", None) or "").strip()
    if not raw:
        return None

    # A recognized close means this is a completed response, not an unfinished trace.
    pattern = think_end_token or CLOSING_TAGS
    if re.search(pattern, raw):
        return None

    text = (getattr(generation, "text", None) or "").strip()
    parser_detected_missing_close = (
        metadata.get("has_think_end_token") is False
        or (not text and bool(metadata.get("reasoning")))
    )
    prompt_left_thinking_open = (
        OPEN_THINKING_PROMPT.search(
            getattr(generation, "prompt", "") or ""
        )
        is not None
    )
    if parser_detected_missing_close or prompt_left_thinking_open:
        return (metadata.get("reasoning") or raw).strip()

    # Low-effort SFT direct answer (or an ordinary non-thinking response).
    return None


class ThinkingHistory:
    """Builds assistant history messages in the shape this model's template wants.

    Used as a context manager because the structured field and the empty-tag
    `history_assistant_wrap` are alternatives, never both: where the template renders the
    field it emits the tags itself and must not also receive a hand-built empty pair. The
    wrap is restored on exit so a model instance shared across tasks is unaffected.
    """

    def __init__(self, model):
        self.model = model
        self.key = resolve_history_key(model)
        self._previous_wrap = None

    def __enter__(self):
        if self.key:
            self._previous_wrap = self.model.history_assistant_wrap
            self.model.history_assistant_wrap = "{content}"
        return self

    def __exit__(self, *exc_info):
        if self.key:
            self.model.history_assistant_wrap = self._previous_wrap
        return False

    def from_generation(self, generation) -> dict:
        """History message for a turn the model just produced, carrying its real trace."""
        unfinished = missing_close_trace(
            generation, getattr(self.model, "think_end_token", None)
        )
        if self.key and unfinished is not None:
            # The raw trace remains generation.text/the scoring fallback, but it is not a
            # visible assistant answer in subsequent model history.
            return {"role": "assistant", "content": "", self.key: unfinished}

        trace, answer = split_trace(generation, getattr(self.model, "think_end_token", None))
        if not self.key:
            # No thinking field: keep the wrapper's own representation (raw trace for thinking
            # checkpoints, post-processed answer otherwise) and let the empty-tag wrap render.
            return {"role": "assistant", "content": self.model.history_content(generation)}
        return {"role": "assistant", "content": answer, self.key: trace}

    def from_content(self, content: str) -> dict:
        """History message for a turn we did not generate (a dataset-scripted assistant turn).

        There is no trace to carry, so the field is empty and the template renders the empty
        tag pair -- the same on-distribution shape the empty-tag wrap produces elsewhere.
        """
        if not self.key:
            return {"role": "assistant", "content": content}
        return {"role": "assistant", "content": content, self.key: ""}

    def message(self, message: dict) -> dict:
        """Pass a whole scripted message through, adding the field to assistant turns."""
        if message.get("role") != "assistant":
            return message
        return {**message, **self.from_content(message.get("content", ""))}
