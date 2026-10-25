"""MT-Bench (FastChat llm_judge) — self-conditioned 2-turn task, English + Arabic.

80 questions x 2 user turns. Faithful FastChat flow: generate turn 1 from [user q1]; then
generate turn 2 from [user q1, assistant answer1, user q2] where answer1 is the model's OWN
(think-stripped) turn-1 answer — self-conditioned, which is exactly what lm-eval cannot do
and why this harness exists. The model wrapper re-wraps answer1 with the model's empty
thinking tags in the turn-2 history (history-assistant wrap) so the history matches the
model's completed-turn format.

Each question emits TWO sample records (turn 1, turn 2) with sequential doc_ids
(2*i, 2*i+1). The doc carries everything a judge prompt needs to pick the FastChat
single-v1 / single-math-v1 / multi-turn prompt variant and rebuild the conversation:
question_id, category, turn (1|2), turns (both user prompts), reference (gpt-4 reference
turns, present for the NEED_REF_CATS categories), and prev_answer (turn-2 rows: the
model's turn-1 answer). target = the reference turn text when available else "" (the
1-10 judge grades against the reference, not an exact target).

MT-Bench-Arabic is the same task over MT_Bench_arabic/ (same file layout).
"""

from .. import Task, read_jsonl, register
from ...thinking import ThinkingHistory


class MTBenchBase(Task):
    METRIC = "judge_score_placeholder"
    BENCH_DIR = None  # subdir under LOCAL_BENCHMARK_PATH, FastChat file layout

    def load_docs(self) -> list:
        bench = self.benchmark_path / self.BENCH_DIR
        questions = sorted(
            read_jsonl(bench / "question.jsonl"), key=lambda q: q["question_id"]
        )
        references = {}
        ref_file = bench / "reference_answer" / "gpt-4.jsonl"
        if ref_file.exists():
            for row in read_jsonl(ref_file):
                references[row["question_id"]] = row["choices"][0]["turns"]
        for question in questions:
            question["reference"] = references.get(question["question_id"])
        return questions[: self.limit] if self.limit else questions

    def run(self, model, cli_gen_kwargs: dict) -> list:
        questions = self.load_docs()
        gen_kwargs = self.merged_gen_kwargs(cli_gen_kwargs)
        print(
            f"[mt_eval] {self.NAME}: {len(questions)} questions x 2 self-conditioned turns",
            flush=True,
        )

        turn1_convs = [
            [{"role": "user", "content": q["turns"][0]}] for q in questions
        ]
        turn1_gens = model.generate_batch(turn1_convs, gen_kwargs)

        # Turn-2 history carries the model's OWN turn-1 output, including its real thinking
        # trace, in whatever shape this template accepts: a structured think field where the
        # template renders one (required by ifm/bbq, which rejects any content-only message),
        # otherwise the wrapper's representation under the empty-tag wrap as before.
        with ThinkingHistory(model) as history:
            turn2_convs = [
                [
                    {"role": "user", "content": q["turns"][0]},
                    history.from_generation(g1),
                    {"role": "user", "content": q["turns"][1]},
                ]
                for q, g1 in zip(questions, turn1_gens)
            ]
            turn2_gens = model.generate_batch(turn2_convs, gen_kwargs)

        samples = []
        for i, (q, g1, g2) in enumerate(zip(questions, turn1_gens, turn2_gens)):
            base_doc = {
                "question_id": q["question_id"],
                "category": q["category"],
                "turns": q["turns"],
                "reference": q["reference"],
            }
            reference = q["reference"] or ["", ""]
            samples.append(
                self.record(
                    doc_id=2 * i,
                    doc={**base_doc, "turn": 1},
                    target=reference[0],
                    generation=g1,
                    gen_kwargs=gen_kwargs,
                )
            )
            samples.append(
                self.record(
                    doc_id=2 * i + 1,
                    doc={**base_doc, "turn": 2, "prev_answer": g1.text},
                    target=reference[1],
                    generation=g2,
                    gen_kwargs=gen_kwargs,
                )
            )
        return samples

    def task_config(self) -> dict:
        config = super().task_config()
        config.update(
            {
                "dataset_path": "json",
                "dataset_kwargs": {
                    "data_files": {"test": f"{self.BENCH_DIR}/question.jsonl"}
                },
                "test_split": "test",
            }
        )
        return config


@register
class MTBench(MTBenchBase):
    NAME = "mt_bench"
    BENCH_DIR = "MT_Bench"


@register
class MTBenchArabic(MTBenchBase):
    NAME = "mt_bench_arabic"
    BENCH_DIR = "MT_Bench_arabic"