"""MultiTurnLM — vLLM wrapper reproducing the jais3 lm-eval fork's generation semantics
for multi-turn, message-list prompting.

The wrapper owns everything model-aware so tasks stay pure orchestration:

* model_args parsing (JSON from the pipeline's helpers.get_model_args, or lm-eval csv k=v);
* chat template application with thinking config (enable_thinking / chat_template_args
  {reasoning_effort}), incl. the fork's TemplateError retry-without-system fallback;
* the history-assistant thinking-tag wrap: a thinking model emits
  "<open_think>...</close_think>ANSWER" but chat templates render PRIOR assistant turns as
  plain content, so multi-turn history is off-distribution unless each prior assistant turn
  is re-wrapped in the model's (empty) thinking tags. Derived from the thinking config:
    reasoning_effort=low  -> "<think_faster>\\n</think_faster>\\n{content}"   (jais-3/xllm)
    reasoning_effort=high -> "<think>\\n</think>\\n{content}"                 (jais-3/xllm)
    otherwise             -> "{content}"                                      (gemma-4 etc.)
  Overridable via model_args history_assistant_wrap.
* sampling faithful to the fork's modify_gen_kwargs: until + decoded eos as string stops,
  skip_special_tokens=False, spaces_between_special_tokens=False, do_sample=False -> temp 0;
* completion metadata identical to the fork's _get_completion_metadata, minus
  answer_before_post_processing (popped into the response, never logged — matches the
  reference sample). Thinking-mode metadata (the +8 keys) is keyed on think_end_token being
  set: the pipeline sets it for every "think" run and never for "instruct" runs, which is
  exactly the split in the reference sample. (The fork keys on `enable_thinking OR
  think_end_token`, which for reasoning_effort=low runs yields resps="" with the whole
  answer stuck in metadata.reasoning — the judge-prompt fallback papers over it. We emit
  the clean form.)
"""

import copy
import json
import re

from .thinking import CLOSING_TAGS as THINKING_CLOSING_TAGS
from .thinking import OPEN_THINKING_PROMPT

# THINKING_CLOSING_TAGS remains this module's compatibility name; the canonical regexes live
# in thinking.py so scoring and history classification cannot drift apart.

_HISTORY_WRAP_BY_EFFORT = {
    "low": "<think_faster>\n</think_faster>\n{content}",
    "high": "<think>\n</think>\n{content}",
}


def _coerce(value: str):
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def simple_parse_args_string(arg_string: str) -> dict:
    """lm-eval style 'k=v,k2=v2' parsing (values coerced like utils.simple_parse_args_string)."""
    parsed = {}
    for part in arg_string.split(","):
        part = part.strip()
        if not part:
            continue
        key, _, value = part.partition("=")
        parsed[key.strip()] = _coerce(value.strip())
    return parsed


def parse_model_args(arg_string: str) -> dict:
    """The pipeline passes model_args as a JSON object (helpers.get_model_args); plain
    lm-eval invocations pass csv k=v. Accept both."""
    arg_string = arg_string.strip()
    if arg_string.startswith("{"):
        return json.loads(arg_string)
    return simple_parse_args_string(arg_string)


def handle_stop_sequences(until, eos):
    # mirrors lm_eval.models.utils.handle_stop_sequences
    if isinstance(until, str):
        until = [until]
    elif until is None:
        until = []
    elif not isinstance(until, list):
        raise ValueError(f"Expected `until` to be Union[str, list] but got {until!r}")
    if eos is not None and eos not in until:
        until.append(eos)
    return until


def postprocess_generated_text(generation: str, stop, think_end_token=None) -> str:
    # mirrors lm_eval.models.utils.postprocess_generated_text
    if stop:
        stop = [stop] if isinstance(stop, str) else stop
        for term in stop:
            if len(term) > 0:
                generation = generation.split(term)[0]
    if think_end_token:
        matches = list(re.finditer(think_end_token, generation))
        if matches:
            generation = generation[matches[-1].end():].lstrip()
    return generation


def postprocess_response_text(
    generation: str,
    prompt: str,
    stop,
    think_end_token,
    chat_template_args: dict,
) -> str:
    """Post-process the response stored in samples and consumed by scorers.

    High-effort generations have already been split by completion metadata.  Low-effort
    SFT/BBQ generations have no configured ``think_end_token``; split them here only when
    the rendered prompt left a thinking block open.  No close means an unfinished trace,
    so the raw text remains available to the existing evaluation fallback.
    """
    low_effort_open_trace = (
        think_end_token is None
        and (chat_template_args or {}).get("reasoning_effort") == "low"
        and OPEN_THINKING_PROMPT.search(prompt or "") is not None
    )
    effective_think_end_token = THINKING_CLOSING_TAGS if low_effort_open_trace else None
    return postprocess_generated_text(generation, stop, effective_think_end_token)


class Generation:
    """One completion: post-processed text (== resps[0][0]), the raw generation, the
    rendered prompt (== arguments.gen_args_0.arg_0) and the metadata record
    (== generation_metadata[0][0])."""

    __slots__ = ("text", "raw", "prompt", "metadata")

    def __init__(self, text, raw, prompt, metadata):
        self.text = text
        self.raw = raw
        self.prompt = prompt
        self.metadata = metadata


class MultiTurnLM:
    """vLLM offline engine + tokenizer, driven by the same model_args the pipeline builds.

    Wrapper-consumed keys (same names/defaults as the fork's VLLM class): pretrained,
    max_model_len/max_length, enable_thinking, chat_template_args, think_end_token,
    log_completion_metadata, max_gen_toks, history_assistant_wrap (ours). Everything else
    (dtype, seed, gpu_memory_utilization, tensor_parallel_size, trust_remote_code,
    attention_backend, max_num_batched_tokens, ...) is forwarded to vllm.LLM, exactly like
    the fork forwards its kwargs.
    """

    def __init__(
        self,
        pretrained: str,
        system_instruction: str = None,
        max_model_len=None,
        max_length=None,
        enable_thinking: bool = True,
        chat_template_args: dict = None,
        think_end_token: str = None,
        log_completion_metadata: bool = True,
        max_gen_toks: int = 256,
        history_assistant_wrap: str = None,
        history_include_thinking=None,
        trust_remote_code: bool = False,
        **vllm_kwargs,
    ):
        from transformers import AutoTokenizer
        from vllm import LLM

        assert max_length is None or max_model_len is None, (
            "Either max_length or max_model_len may be provided, but not both"
        )
        self.pretrained = pretrained
        self.system_instruction = system_instruction
        self.think_end_token = think_end_token
        self.log_completion_metadata = log_completion_metadata
        self.max_gen_toks = max_gen_toks
        self.chat_template_args = dict(chat_template_args or {})
        # fork behavior: enable_thinking inside chat_template_args wins over the top-level arg
        self.enable_thinking = self.chat_template_args.pop("enable_thinking", enable_thinking)
        self._max_length = max_model_len if max_model_len is not None else max_length

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

        self.model_args = {
            "model": pretrained,
            "trust_remote_code": trust_remote_code,
            "max_model_len": int(self._max_length) if self._max_length else None,
            **vllm_kwargs,
        }
        self.llm = LLM(**self.model_args)
        if self._max_length is None:
            self._max_length = self.llm.llm_engine.model_config.max_model_len

        effort = self.chat_template_args.get("reasoning_effort")
        if history_assistant_wrap is not None:
            # allow literal \n when the wrap comes through a csv model_args string
            self.history_assistant_wrap = history_assistant_wrap.replace("\\n", "\n")
        else:
            self.history_assistant_wrap = _HISTORY_WRAP_BY_EFFORT.get(effort, "{content}")
        # For thinking CHECKPOINTS (reasoning_effort set), carry the model's REAL prior-turn
        # thinking trace into multi-turn history (its raw "<think>…</think>ANSWER") instead of
        # the think-stripped answer wrapped in EMPTY tags. Default on when reasoning_effort is
        # set (the xllm/jais-3 checkpoints); gemma (no reasoning_effort) keeps plain history.
        if history_include_thinking is None:
            history_include_thinking = effort is not None
        self.history_include_thinking = bool(history_include_thinking)
        print(
            f"[mt_eval] model={pretrained} enable_thinking={self.enable_thinking} "
            f"chat_template_args={self.chat_template_args} "
            f"think_end_token={'set' if self.think_end_token else None} "
            f"history_assistant_wrap={self.history_assistant_wrap!r} "
            f"history_include_thinking={self.history_include_thinking}",
            flush=True,
        )

    # ------------------------------------------------------------------ properties

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    def model_info(self) -> dict:
        """Fields the aggregated results_*.json logs about the model/tokenizer."""

        def tok_pair(token):
            if token is None:
                return None
            return [token, str(self.tokenizer.convert_tokens_to_ids(token))]

        return {
            "model_name": self.pretrained,
            "tokenizer_pad_token": tok_pair(self.tokenizer.pad_token),
            "tokenizer_eos_token": tok_pair(self.tokenizer.eos_token),
            "tokenizer_bos_token": tok_pair(self.tokenizer.bos_token),
            "eot_token_id": self.eot_token_id,
            "max_length": self.max_length,
            "chat_template": self.tokenizer.chat_template
            if isinstance(self.tokenizer.chat_template, str)
            else None,
        }

    # ------------------------------------------------------------------ rendering

    def _wrap_assistant(self, content: str) -> str:
        """Apply the empty-tag history wrap to a prior assistant turn — but IDEMPOTENTLY:
        if the content already carries a real thinking trace (starts with a think-open tag, as
        history_content returns for thinking checkpoints), leave it untouched so we don't inject
        empty tags in front of a real "<think>…</think>ANSWER". Plain answers (gemma / the
        dataset-provided MultiChallenge history) still get the empty-tag wrap as before."""
        wrap = self.history_assistant_wrap
        if wrap == "{content}":
            return content
        if (content or "").lstrip().startswith(("<think", "<|think", "<channel", "<ifm")):
            return content
        return wrap.format(content=content)

    def apply_chat_template(self, messages: list, add_generation_prompt: bool = True) -> str:
        """Render a message list; every PRIOR assistant turn goes through the
        history-assistant wrap first (see module docstring)."""
        if self.system_instruction:
            messages = [
                {"role": "system", "content": self.system_instruction},
                *messages,
            ]
        wrapped = [
            {**m, "content": self._wrap_assistant(m["content"])}
            if m["role"] == "assistant"
            else m
            for m in messages
        ]
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
            enable_thinking=self.enable_thinking,
            **self.chat_template_args,
        )
        try:
            return self.tokenizer.apply_chat_template(wrapped, **kwargs)
        except Exception:
            # fork fallback: some templates reject a system role
            return self.tokenizer.apply_chat_template(
                [m for m in wrapped if m["role"] != "system"], **kwargs
            )

    def history_content(self, gen) -> str:
        """Assistant-history representation of a completed generation, for self-conditioned
        multi-turn tasks (MT-Bench). For thinking CHECKPOINTS (history_include_thinking) return
        the RAW turn output — the model's real "<think>…</think>ANSWER" — so a subsequent turn
        sees the prior turn's thinking trace (the idempotent wrap leaves it intact). Otherwise
        return the post-processed answer (gemma / non-thinking), which the empty-tag wrap then
        renders on-distribution as before."""
        if self.history_include_thinking:
            return (getattr(gen, "raw", None) or gen.text or "").strip()
        return gen.text

    def tok_encode(self, prompt: str) -> list:
        # chat-templated strings already carry their special tokens (e.g. gemma's <bos>)
        return self.tokenizer(prompt, add_special_tokens=False).input_ids

    # ------------------------------------------------------------------ sampling

    def _modify_gen_kwargs(self, gen_kwargs: dict):
        """Mirror of the fork's modify_gen_kwargs: returns (SamplingParams kwargs, until,
        max_gen_toks). eos is the decoded eot token, appended to the stop strings; vLLM
        additionally stops on the model's generation_config eos ids by itself (that is how
        gemma's <turn|> id 106 shows up as an int stop_reason, same as in lm-eval runs)."""
        kwargs = copy.deepcopy(gen_kwargs)
        eos = (
            self.tokenizer.decode([self.eot_token_id])
            if self.eot_token_id is not None
            else None
        )
        until = handle_stop_sequences(kwargs.pop("until", None), eos)
        if "<|END_RESPONSE|>" not in until:
            until.append("<|END_RESPONSE|>")
        max_gen_toks = int(kwargs.pop("max_gen_toks", self.max_gen_toks))
        do_sample = kwargs.pop("do_sample", None)
        if do_sample is False:
            kwargs["temperature"] = 0.0
        kwargs = {
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        } | kwargs
        return kwargs, until, max_gen_toks

    def _fit_window(self, token_ids: list, max_gen_toks: int):
        """The pipeline passes max_gen_toks == max_model_len, where lm-eval's context slice
        arithmetic is a no-op and vLLM clamps max_tokens to the remaining window. Implement
        that effective behavior directly: keep the context (left-truncate only if it alone
        overflows), cap max_tokens to what fits."""
        if len(token_ids) + max_gen_toks <= self.max_length:
            return token_ids, max_gen_toks
        if len(token_ids) >= self.max_length:
            print(
                f"[mt_eval] WARNING: context ({len(token_ids)} toks) >= max_length "
                f"({self.max_length}); left-truncating",
                flush=True,
            )
            token_ids = token_ids[-(self.max_length - 1):]
        return token_ids, min(max_gen_toks, self.max_length - len(token_ids))

    # ------------------------------------------------------------------ metadata

    @staticmethod
    def _find_last_subsequence(sequence: list, subsequence: list):
        if not subsequence or len(subsequence) > len(sequence):
            return None
        for start in range(len(sequence) - len(subsequence), -1, -1):
            if sequence[start:start + len(subsequence)] == subsequence:
                return start
        return None

    def _encode_metadata_text(self, text: str) -> list:
        if not text:
            return []
        token_ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return list(token_ids)

    def _get_completion_metadata(self, completion) -> dict:
        """Fork's _get_completion_metadata with answer_before_post_processing carried
        separately (the caller pops it into the response; it is never logged — the
        reference sample has no such key)."""
        raw_text = completion.text
        token_ids = list(getattr(completion, "token_ids", None) or [])

        base_metadata = {
            "finish_reason": getattr(completion, "finish_reason", None),
            "stop_reason": getattr(completion, "stop_reason", None),
            "generated_token_count": len(token_ids),
            "word_count": len(raw_text.split()),
        }

        if self.think_end_token is None:
            return base_metadata | {"answer_before_post_processing": raw_text}

        think_end_match = None
        reasoning = raw_text
        answer = ""
        reasoning_token_count = len(token_ids)
        answer_token_count = 0

        matches = list(re.finditer(self.think_end_token, raw_text))
        if matches:
            think_end_match = matches[-1]
            reasoning = raw_text[: think_end_match.start()]
            answer = raw_text[think_end_match.end():].lstrip()

            delimiter_token_ids = self._encode_metadata_text(think_end_match.group(0))
            delimiter_start = self._find_last_subsequence(token_ids, delimiter_token_ids)
            if delimiter_start is None:
                reasoning_token_count = len(self._encode_metadata_text(reasoning))
                answer_token_count = len(self._encode_metadata_text(answer))
            else:
                delimiter_end = delimiter_start + len(delimiter_token_ids)
                reasoning_token_count = delimiter_start
                answer_token_count = len(token_ids) - delimiter_end

        return base_metadata | {
            "has_think_end_token": think_end_match is not None,
            "think_end_token": self.think_end_token,
            "matched_think_end_token": (
                think_end_match.group(0) if think_end_match else None
            ),
            "reasoning": reasoning,
            "answer_before_post_processing": answer,
            "reasoning_word_count": len(reasoning.split()),
            "answer_word_count": len(answer.split()),
            "reasoning_token_count": reasoning_token_count,
            "answer_token_count": answer_token_count,
        }

    # ------------------------------------------------------------------ generation

    def generate_batch(self, conversations: list, gen_kwargs: dict) -> list:
        """Generate one completion per message-list, batched through vLLM's continuous
        batching. Returns a Generation per conversation, in order."""
        from vllm import SamplingParams

        prompts = [self.apply_chat_template(conv) for conv in conversations]
        kwargs, until, max_gen_toks = self._modify_gen_kwargs(gen_kwargs)

        inputs, sampling_params = [], []
        for prompt in prompts:
            token_ids, fitted_max = self._fit_window(self.tok_encode(prompt), max_gen_toks)
            inputs.append({"prompt_token_ids": token_ids})
            sampling_params.append(
                SamplingParams(max_tokens=fitted_max, stop=until, **kwargs)
            )

        outputs = self.llm.generate(inputs, sampling_params)

        generations = []
        for prompt, output in zip(prompts, outputs):
            completion = output.outputs[0]
            metadata = self._get_completion_metadata(completion)
            text = metadata.pop("answer_before_post_processing")
            # Secondary stop cut, post-hoc.  Also split valid low-effort BBQ traces, whose
            # model args intentionally have no think_end_token, while leaving low-effort SFT
            # direct answers and missing-close fallbacks unchanged.
            text = postprocess_response_text(
                text,
                prompt,
                until,
                self.think_end_token,
                self.chat_template_args,
            )
            generations.append(
                Generation(text=text, raw=completion.text, prompt=prompt, metadata=metadata)
            )
        return generations
