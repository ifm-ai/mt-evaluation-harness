"""AraGen 3C3H generation, per the official generate-oma.py: Round 0 answers Q1; Round 1 answers Q2
given the GOLD A1 as context; Round 2 answers Q1 then Q2 self-conditioned on its own A1. One sample
per question, Round 2 carrying its follow-up in doc["model_answer_2"] so the judge scores both."""

from .. import Task, read_jsonl, register
from ...thinking import ThinkingHistory


class AraGenBase(Task):
    METRIC = "judge_score_placeholder"
    BENCH_FILE = None  # jsonl under LOCAL_BENCHMARK_PATH/aragen/

    def load_docs(self) -> list:
        rows = read_jsonl(self.benchmark_path / "aragen" / self.BENCH_FILE)
        return rows[: self.limit] if self.limit else rows

    def run(self, model, cli_gen_kwargs: dict) -> list:
        rows = self.load_docs()
        gen_kwargs = self.merged_gen_kwargs(cli_gen_kwargs)
        print(f"[mt_eval] {self.NAME}: {len(rows)} questions (rounds "
              f"{sorted({r.get('Round', 0) for r in rows})})", flush=True)

        # Assistant turns go through ThinkingHistory: ifm/bbq templates reject a content-only
        # assistant message, and Round 2 must carry the model's real trace into its second turn.
        with ThinkingHistory(model) as history:
            primary_convs, meta = [], []
            for i, r in enumerate(rows):
                rnd = r.get("Round", 0)
                q1 = r.get("Question 1") or ""
                if rnd == 1:
                    conv = [
                        {"role": "user", "content": q1},
                        history.from_content(r.get("Answer 1") or ""),   # gold A1 as context
                        {"role": "user", "content": r.get("Question 2") or ""},
                    ]
                else:  # Round 0 and Round 2 both start by answering Q1
                    conv = [{"role": "user", "content": q1}]
                primary_convs.append(conv)
                meta.append((i, r, rnd))

            primary_gens = model.generate_batch(primary_convs, gen_kwargs)

            r2_idx = [i for i, (_, _, rnd) in enumerate(meta) if rnd == 2]
            r2_convs = [
                [
                    {"role": "user", "content": rows[meta[i][0]].get("Question 1") or ""},
                    history.from_generation(primary_gens[i]),
                    {"role": "user", "content": rows[meta[i][0]].get("Question 2") or ""},
                ]
                for i in r2_idx
            ]
            r2_gens = model.generate_batch(r2_convs, gen_kwargs) if r2_convs else []
            r2_gen_by_i = dict(zip(r2_idx, r2_gens))

        samples = []
        for doc_id, (i, r, rnd) in enumerate(meta):
            doc = {k: r.get(k) for k in (
                "SN", "Round", "Task", "Category", "Sub-Category", "Source",
                "Question 1", "Answer 1", "Question 2", "Answer 2")}
            if rnd == 2:
                doc["model_answer_2"] = r2_gen_by_i[i].text
            samples.append(self.record(
                doc_id=doc_id, doc=doc,
                target=r.get("Answer 2") if rnd in (1, 2) else r.get("Answer 1"),
                generation=primary_gens[i], gen_kwargs=gen_kwargs))
        return samples

    def task_config(self) -> dict:
        config = super().task_config()
        config.update({
            "dataset_path": "json",
            "dataset_kwargs": {"data_files": {"test": f"aragen/{self.BENCH_FILE}"}},
            "test_split": "test",
        })
        return config


@register
class AraGen1224(AraGenBase):
    NAME = "aragen_12_24"
    BENCH_FILE = "aragen_12_24.jsonl"


@register
class AraGen0325(AraGenBase):
    NAME = "aragen_03_25"
    BENCH_FILE = "aragen_03_25.jsonl"
