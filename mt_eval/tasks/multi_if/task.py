import json

from .. import Task, read_jsonl, register
from ...thinking import ThinkingHistory, missing_close_trace, split_trace
from .utils import import_ifeval, print_summary, results_entry, score_records


def _parse_conversation(row: dict) -> dict:
    """Parse one source row into a turn-aligned conversation dict.

    Present turns are those with a non-None prompt and form a contiguous prefix 1..n.
    Returns user_prompts / instruction_id_lists / kwargs_lists, each length n (= num_turns).
    """
    user_prompts, instruction_id_lists, kwargs_lists = [], [], []
    for turn in (1, 2, 3):
        raw_prompt = row.get(f"turn_{turn}_prompt")
        if raw_prompt is None:
            break  # turns are a contiguous prefix; nothing follows an absent turn
        user_prompts.append(json.loads(raw_prompt)["content"])
        instruction_id_lists.append(json.loads(row[f"turn_{turn}_instruction_id_list"]))
        kwargs_lists.append(
            [json.loads(kw) for kw in json.loads(row[f"turn_{turn}_kwargs"])]
        )
    return {
        "key": row["key"],
        "num_turns": len(user_prompts),
        "user_prompts": user_prompts,
        "instruction_id_lists": instruction_id_lists,
        "kwargs_lists": kwargs_lists,
    }


@register
class MultiIF(Task):
    NAME = "multi_if"
    # A real metric, not a placeholder: Multi-IF is verified programmatically, so the score is
    # computed here at generation time (see utils.py) exactly as lm-eval's own `ifeval` does.
    # There is no judge and no JUDGE_REGISTRY entry.
    METRIC = "multi_if_accuracy"
    DATA_FILE = "MultiIF/benchmark_questions_english.jsonl"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._metrics = None
        self._diagnostics = None

    def load_docs(self) -> list:
        rows = read_jsonl(self.benchmark_path / self.DATA_FILE)
        rows = rows[: self.limit] if self.limit else rows
        return [_parse_conversation(row) for row in rows]

    def run(self, model, cli_gen_kwargs: dict) -> list:
        convs = self.load_docs()
        gen_kwargs = self.merged_gen_kwargs(cli_gen_kwargs)
        max_turns = max((c["num_turns"] for c in convs), default=0)
        print(
            f"[mt_eval] {self.NAME}: {len(convs)} conversations, up to {max_turns} "
            f"self-conditioned turns",
            flush=True,
        )

        # Running chat history per conversation (grows as [u1, a1, u2, a2, ...]); the model's
        # own reasoning is fed back in as a structured think field on the assistant turn.
        histories = [[] for _ in convs]
        # generations[conv_idx][turn_idx] -> Generation, for present turns only
        generations = [[None] * c["num_turns"] for c in convs]

        # History assistant turns go through ThinkingHistory, the one path shared with
        # mt_bench/multichallenge: it picks the template's thinking field (ifm/bbq REQUIRE one),
        # neutralises the empty-tag wrap while that field is in use, and restores it on exit.
        with ThinkingHistory(model) as history:
            for turn_idx in range(max_turns):
                active = [i for i, c in enumerate(convs) if turn_idx < c["num_turns"]]
                batch = []
                for i in active:
                    histories[i].append(
                        {"role": "user", "content": convs[i]["user_prompts"][turn_idx]}
                    )
                    batch.append(histories[i])
                print(
                    f"[mt_eval] {self.NAME}: turn {turn_idx + 1} -> {len(active)} conversations",
                    flush=True,
                )
                outputs = model.generate_batch(batch, gen_kwargs)
                for i, gen in zip(active, outputs):
                    # Build the history message BEFORE touching the generation: it reads the
                    # untouched text/raw/metadata to decide trace-vs-answer.
                    histories[i].append(history.from_generation(gen))
                    # What the IFEval checkers verify: the final answer, or — when thinking
                    # never closed — the raw trace as the fallback response (policy rule 3),
                    # which also shows up as `no_close_tag` in the scoring diagnostics.
                    unfinished = missing_close_trace(gen, getattr(model, "think_end_token", None))
                    if unfinished is not None:
                        gen.text = unfinished
                    else:
                        _, answer = split_trace(gen, getattr(model, "think_end_token", None))
                        gen.text = answer
                    generations[i][turn_idx] = gen

        samples, doc_id = [], 0
        for conv, conv_gens in zip(convs, generations):
            for turn_idx in range(conv["num_turns"]):
                doc = {
                    "key": conv["key"],
                    "language": "English",
                    "turn": turn_idx + 1,
                    "num_turns": conv["num_turns"],
                    "prompt": conv["user_prompts"][turn_idx],
                    # cumulative through this turn; checked against THIS turn's response
                    "instruction_id_list": conv["instruction_id_lists"][turn_idx],
                    "kwargs": conv["kwargs_lists"][turn_idx],
                }
                samples.append(
                    self.record(
                        doc_id=doc_id,
                        doc=doc,
                        target="",  # no reference; instructions verified programmatically
                        generation=conv_gens[turn_idx],
                        gen_kwargs=gen_kwargs,
                    )
                )
                doc_id += 1

        self.score(samples)
        return samples

    def score(self, samples: list) -> None:
        """Verify every response against its instructions and fill in the real metric.

        Runs inline after generation because the check is deterministic and CPU-only — the
        pipeline's separate scoring stage (judge_vllm) exists for LLM judges, which this task
        does not use. Each record's metric value is set here; the aggregate goes out through
        results_entry() into results_*.json.
        """
        process_results, agg_inst_level_acc = import_ifeval()
        self._metrics, self._diagnostics = score_records(
            samples, process_results, agg_inst_level_acc, annotate=True
        )
        print_summary(self.NAME, self._metrics, self._diagnostics, len(samples))

    def results_entry(self, n_samples: int) -> dict:
        if self._metrics is None:
            # scored during run(); a bare results_entry() call has nothing to report
            return super().results_entry(n_samples)
        return results_entry(self.NAME, self._metrics, self._diagnostics, n_samples)

    def task_config(self) -> dict:
        config = super().task_config()
        config.update(
            {
                "dataset_path": "json",
                "dataset_kwargs": {"data_files": {"test": self.DATA_FILE}},
                "test_split": "test",
            }
        )
        return config
