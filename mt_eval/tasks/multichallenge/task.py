"""MultiChallenge (arXiv:2501.17399) — faithful multi-turn task.

Each of the 273 examples is a scripted conversation (user+assistant turns, up to ~19)
ending on a USER turn; the model generates ONLY the final assistant turn given that real
multi-turn history (one generation per doc, not self-conditioned). This is the faithful
replacement for the flattened-transcript lm-eval task of the same name: instead of
rendering the conversation as a "User:/Assistant:" transcript inside a single user
message, the history is passed as actual chat-template turns (with the model wrapper's
history-assistant thinking-tag wrap keeping prior turns on-distribution).

Downstream is unchanged: doc/target carry the same fields as the lm-eval task
(doc=raw dataset row, target=PASS_CRITERIA), so judge_registry's multichallenge prompt builder
reads doc["TARGET_QUESTION"] + resps[0][0] as-is, and judge_vllm routes via that registry entry
and writes acc,none = pass-rate.
"""

from .. import Task, read_jsonl, register
from ...thinking import ThinkingHistory


@register
class MultiChallenge(Task):
    NAME = "multichallenge"
    METRIC = "pass_rate_placeholder"  # same placeholder as the lm-eval task's metric fn
    DATA_FILE = "MultiChallenge/benchmark_questions.jsonl"

    def load_docs(self) -> list:
        docs = read_jsonl(self.benchmark_path / self.DATA_FILE)
        return docs[: self.limit] if self.limit else docs

    def run(self, model, cli_gen_kwargs: dict) -> list:
        docs = self.load_docs()
        gen_kwargs = self.merged_gen_kwargs(cli_gen_kwargs)
        print(f"[mt_eval] {self.NAME}: {len(docs)} conversations", flush=True)

        # The history is dataset-scripted, so its assistant turns carry no thinking trace of
        # the model's own. Templates that take a thinking field still need it present (ifm/bbq
        # raises without it); an empty one renders the empty tag pair, which is the same
        # on-distribution shape the empty-tag wrap produces for the other templates.
        with ThinkingHistory(model) as history:
            conversations = [
                [history.message(turn) for turn in doc["CONVERSATION"]] for doc in docs
            ]
            generations = model.generate_batch(conversations, gen_kwargs)

        return [
            self.record(
                doc_id=doc_id,
                doc=doc,
                target=doc["PASS_CRITERIA"],
                generation=generation,
                gen_kwargs=gen_kwargs,
            )
            for doc_id, (doc, generation) in enumerate(zip(docs, generations))
        ]

    def task_config(self) -> dict:
        config = super().task_config()
        config.update(
            {
                "dataset_path": "json",
                "dataset_kwargs": {"data_files": {"test": self.DATA_FILE}},
                "test_split": "test",
                "doc_to_target": "{{PASS_CRITERIA}}",
            }
        )
        return config