"""mt_eval — multi-turn, LLM-judged benchmark generation harness.

A parallel engine to lm-evaluation-harness for benchmarks that need model-aware
multi-turn generation (self-conditioned turns, per-model thinking-tag history). CLI and
outputs are lm-eval-compatible: the jais-3-evals pipeline stages downstream of generation
(judge_vllm.py, merge_results.py, eval_status.py, dashboard) consume them unchanged.
"""

__version__ = "0.1.0"
