"""Programmatic Multi-IF (arXiv:2410.15553) scoring — NO LLM judge.

Multi-IF extends IFEval: each turn's response is checked against verifiable instructions by
rule/script, exactly as in the paper (§4.1). Because the metric is deterministic, it is
computed **in-task** at generation time and written straight into results_*.json — the same
place and shape lm-eval's own `ifeval` puts `prompt_level_strict_acc,none` etc. Nothing here
runs at judging time; `multi_if` is deliberately absent from the pipeline's JUDGE_REGISTRY.

We reuse the canonical IFEval checkers verbatim from the lm-eval fork
(lm_eval/tasks/ifeval/utils.py: test_instruction_following_strict/loose via process_results)
so strict/loose semantics, the prompt injection, and the 8 loose response variants match the
reference implementation.

Metrics (per turn, then averaged over turns), matching Table 1:
  inst_level_strict_acc    micro over instructions: fraction of individual instructions passed
  prompt_level_strict_acc  conversation-level: fraction of turns where ALL instructions passed
  inst_level_loose_acc / prompt_level_loose_acc   loose (boilerplate-stripped) variants
  multi_if_accuracy        = mean of the four; headline acc,none = mean over turns

Because instructions are cumulative in the data, a turn-t response is verified against every
instruction from turns 1..t — so a forgotten early constraint fails at the later turn, which is
the degradation the benchmark measures.

Thinking-model safety: a response with no closing think tag can end up empty and then fails
everything. That is faithful (a non-answer is a failure), but usually indicates truncation
rather than disobedience, so we surface empty / length-truncated / missing-close rates per turn
and warn loudly above a threshold — a token-budget problem shows up as a diagnostic instead of
a silently deflated score.
"""

import logging
import math
from collections import defaultdict
from contextlib import contextmanager

# warn if more than this fraction of a turn's answers are empty / length-truncated
INVALID_RATE_THRESHOLD = 0.02

METRIC_NAMES = (
    "inst_level_strict_acc",
    "prompt_level_strict_acc",
    "inst_level_loose_acc",
    "prompt_level_loose_acc",
    "multi_if_accuracy",
)


def import_ifeval():
    """Import process_results + agg_inst_level_acc from the lm-eval fork.

    A plain import: `lm_eval` is pip-installed (editable) into every eval venv, so it resolves
    to the fork the pipeline actually runs. Do NOT reintroduce a sys.path insert pointing at a
    checkout — that would silently shadow the installed copy and mix two lm-eval versions.
    """
    try:
        from lm_eval.tasks.ifeval.utils import agg_inst_level_acc, process_results
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"Could not import the IFEval checkers ({exc}). Run this in an eval venv, which "
            f"has lm_eval installed along with the checkers' deps (langdetect, immutabledict, "
            f"nltk) — see mt_eval's `ifeval` extra."
        )
    return process_results, agg_inst_level_acc


@contextmanager
def _quiet_langdetect():
    """Silence the language checker's expected ERROR logs for the scoring loop only.

    The canonical IFEval language checker logs LangDetectException cases (an empty string, for
    one) at ERROR level before treating them as non-fatal. Over a few thousand records that
    buries the real job log, so raise the threshold here and restore it straight after.
    """
    previous = logging.root.manager.disable
    logging.disable(logging.ERROR)
    try:
        yield
    finally:
        logging.disable(previous)


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _stderr(values):
    """Standard error of the mean, as lm-eval reports alongside each accuracy."""
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def per_sample_score(out: dict) -> dict:
    """The four IFEval accuracies for one record plus a scalar roll-up (mean of the four, same
    definition as the per-turn multi_if_accuracy). inst-level are kept as the raw
    per-instruction bool lists so nothing is lost."""
    strict_prompt = bool(out["prompt_level_strict_acc"])
    loose_prompt = bool(out["prompt_level_loose_acc"])
    strict_insts = [bool(x) for x in out["inst_level_strict_acc"]]
    loose_insts = [bool(x) for x in out["inst_level_loose_acc"]]
    strict_inst_mean = _mean(strict_insts)
    loose_inst_mean = _mean(loose_insts)
    return {
        "prompt_level_strict_acc": strict_prompt,
        "prompt_level_loose_acc": loose_prompt,
        "inst_level_strict_acc": strict_insts,
        "inst_level_loose_acc": loose_insts,
        "inst_level_strict_acc_mean": strict_inst_mean,
        "inst_level_loose_acc_mean": loose_inst_mean,
        "multi_if_accuracy": _mean(
            [strict_prompt, loose_prompt, strict_inst_mean, loose_inst_mean]
        ),
    }


def score_records(records: list, process_results, agg_inst_level_acc, annotate: bool = True):
    """Score samples_multi_if records. Returns (metrics, diagnostics), both keyed by turn plus
    an overall roll-up. Pure and unit-testable — inject the two lm-eval callables.

    With annotate=True (the default in-task path) each record is filled in place: its metric
    key (rec['metrics'][0]) gets the scalar multi_if_accuracy instead of the 0.0 placeholder,
    and a 'multi_if_score' dict carries the full per-record breakdown.
    """
    prompt_strict = defaultdict(list)   # turn -> [bool per conversation-turn]
    prompt_loose = defaultdict(list)
    inst_strict = defaultdict(list)     # turn -> [ [bool per instruction], ... ]
    inst_loose = defaultdict(list)
    scalars = []                        # per-record multi_if_accuracy, for the headline stderr
    diag = defaultdict(lambda: {"n": 0, "empty": 0, "length_truncated": 0, "no_close_tag": 0})

    with _quiet_langdetect():
        for rec in records:
            doc = rec["doc"]
            turn = doc["turn"]
            response = rec["resps"][0][0]
            out = process_results(doc, [response])
            prompt_strict[turn].append(bool(out["prompt_level_strict_acc"]))
            prompt_loose[turn].append(bool(out["prompt_level_loose_acc"]))
            inst_strict[turn].append(list(out["inst_level_strict_acc"]))
            inst_loose[turn].append(list(out["inst_level_loose_acc"]))

            score = per_sample_score(out)
            scalars.append(score["multi_if_accuracy"])
            if annotate:
                rec["multi_if_score"] = score
                metric_keys = rec.get("metrics") or []
                if metric_keys:
                    rec[metric_keys[0]] = score["multi_if_accuracy"]

            # generation-quality diagnostics (thinking-model truncation etc.)
            meta = (rec.get("generation_metadata") or [[{}]])[0][0] or {}
            counters = diag[turn]
            counters["n"] += 1
            if not response.strip():
                counters["empty"] += 1
            if meta.get("finish_reason") == "length":
                counters["length_truncated"] += 1
            if meta.get("has_think_end_token") is False:  # False (not None) => think run, no close
                counters["no_close_tag"] += 1

    turns = sorted(prompt_strict)
    metrics = {"per_turn": {}}
    for turn in turns:
        four = {
            "inst_level_strict_acc": agg_inst_level_acc(inst_strict[turn]),
            "prompt_level_strict_acc": _mean(prompt_strict[turn]),
            "inst_level_loose_acc": agg_inst_level_acc(inst_loose[turn]),
            "prompt_level_loose_acc": _mean(prompt_loose[turn]),
        }
        four["multi_if_accuracy"] = _mean(list(four.values()))
        four["n"] = len(prompt_strict[turn])
        metrics["per_turn"][turn] = four

    # overall = mean over turns of each metric (matches the paper's average-across-turns)
    metrics["overall"] = {
        name: _mean([metrics["per_turn"][t][name] for t in turns]) for name in METRIC_NAMES
    }
    metrics["stderr"] = _stderr(scalars)
    diagnostics = {t: dict(diag[t]) for t in turns}
    return metrics, diagnostics


def results_entry(name: str, metrics: dict, diagnostics: dict, n_records: int) -> dict:
    """The results[<task>] block. Per-turn + overall keys use the ',none' filter suffix like
    every other task; headline acc,none = overall multi_if_accuracy."""
    entry = {
        "name": name,
        "alias": name,
        "sample_len": n_records,
        "acc,none": metrics["overall"]["multi_if_accuracy"],  # headline (0-1, dashboard)
        "acc_stderr,none": metrics["stderr"],
    }
    for metric_name, value in metrics["overall"].items():
        entry[f"{metric_name},none"] = value
    for turn, four in metrics["per_turn"].items():
        for metric_name in METRIC_NAMES:
            entry[f"{metric_name}_turn{turn},none"] = four[metric_name]
    # diagnostics recorded WITHOUT the ',none' suffix so dashboards don't read them as scores
    entry["multi_if_diagnostics"] = diagnostics
    return entry


def print_summary(name: str, metrics: dict, diagnostics: dict, n_records: int):
    """Per-turn table + generation-quality warnings, into the generation job's log."""
    print(f"\n[mt_eval] === {name}: {n_records} records ===")
    print(
        f"{'turn':>4} {'n':>5} {'inst_str':>9} {'prmpt_str':>9} {'inst_lse':>9} "
        f"{'prmpt_lse':>9} {'turn_acc':>9}"
    )
    for turn, four in metrics["per_turn"].items():
        print(
            f"{turn:>4} {four['n']:>5} {four['inst_level_strict_acc']:>9.4f} "
            f"{four['prompt_level_strict_acc']:>9.4f} {four['inst_level_loose_acc']:>9.4f} "
            f"{four['prompt_level_loose_acc']:>9.4f} {four['multi_if_accuracy']:>9.4f}"
        )
    overall = metrics["overall"]
    print(
        f"{'ALL':>4} {n_records:>5} {overall['inst_level_strict_acc']:>9.4f} "
        f"{overall['prompt_level_strict_acc']:>9.4f} {overall['inst_level_loose_acc']:>9.4f} "
        f"{overall['prompt_level_loose_acc']:>9.4f} {overall['multi_if_accuracy']:>9.4f}"
    )
    print(f"  >> headline Multi-IF accuracy (acc,none) = {overall['multi_if_accuracy']:.4f}")

    warned = False
    for turn, counters in diagnostics.items():
        total = counters["n"] or 1
        for label, key in (
            ("empty answers", "empty"),
            ("length-truncated", "length_truncated"),
            ("no close-tag", "no_close_tag"),
        ):
            rate = counters[key] / total
            if counters[key] and rate > INVALID_RATE_THRESHOLD:
                warned = True
                print(
                    f"  !! WARNING turn {turn}: {counters[key]}/{counters['n']} ({rate:.1%}) "
                    f"{label} — likely token-budget/truncation; raise max_gen_toks. "
                    f"These count as failures and deflate the score."
                )
    if not warned:
        print("  generation-quality checks OK (no significant empty/truncated answers)")
