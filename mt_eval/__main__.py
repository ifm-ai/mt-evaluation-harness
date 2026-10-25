"""CLI — mirrors the lm-eval invocation the pipeline builds (helpers.create_lm_eval_cmd),
so the pipeline can route a multi-turn task here by swapping the executable only:

    python -m mt_eval \
        --model vllm \
        --model_args '{"pretrained": ..., "enable_thinking": ..., ...}' \
        --gen_kwargs 'temperature=1.0,do_sample=True,top_p=0.95,...,max_gen_toks=32768' \
        --tasks mt_bench,mt_bench_arabic,multichallenge \
        --output_path <ckpt_results_dir>/results.json \
        --apply_chat_template --log_samples --seed 42 [--limit N]

Unused-but-accepted lm-eval flags (--batch_size, --device, --verbosity,
--confirm_run_unsafe_code, ...) are recorded in the results config block and otherwise
ignored: vLLM continuous batching handles scheduling, and multi-turn tasks are always
chat-templated.
"""

import argparse
import json
import time

from .io import OutputWriter, build_results_dict, compute_task_hash
from .model import MultiTurnLM, parse_model_args, simple_parse_args_string
from .tasks import TASK_REGISTRY, get_task


def parse_args():
    parser = argparse.ArgumentParser(
        prog="mt_eval", description="Multi-turn, LLM-judged benchmark generation harness"
    )
    parser.add_argument("--model", default="vllm", choices=["vllm"])
    parser.add_argument(
        "--model_args",
        required=True,
        help="JSON object (as built by the pipeline's helpers.get_model_args) or lm-eval csv k=v",
    )
    parser.add_argument(
        "--gen_kwargs", default="", help="lm-eval csv k=v, e.g. 'temperature=1.0,do_sample=True'"
    )
    parser.add_argument(
        "--tasks",
        required=True,
        help=f"comma-separated; available: {','.join(sorted(TASK_REGISTRY))}",
    )
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--benchmark_path",
        default=None,
        help="override LOCAL_BENCHMARK_PATH (default /jais3/data/jais_2/benchmarks)",
    )
    # lm-eval compatibility flags
    parser.add_argument("--apply_chat_template", action="store_true")
    parser.add_argument("--system_instruction", type=str, default=None)
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--confirm_run_unsafe_code", action="store_true")
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--device", default=None)
    parser.add_argument("--verbosity", default="INFO")
    return parser.parse_args()


def main():
    args = parse_args()
    start_unix = time.time()
    start = time.perf_counter()

    model_args = parse_model_args(args.model_args)
    gen_kwargs = simple_parse_args_string(args.gen_kwargs) if args.gen_kwargs else {}
    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for name in task_names:
        if name not in TASK_REGISTRY:
            raise SystemExit(
                f"Unknown task {name!r}. Available: {sorted(TASK_REGISTRY)}"
            )
    if not args.apply_chat_template:
        print(
            "[mt_eval] NOTE: multi-turn tasks are always chat-templated "
            "(--apply_chat_template implied)",
            flush=True,
        )

    # --trust_remote_code / --seed behave like lm-eval: folded into the model args
    if args.trust_remote_code:
        model_args["trust_remote_code"] = True
    model_args.setdefault("seed", args.seed)

    model = MultiTurnLM(system_instruction=args.system_instruction, **model_args)

    samples_by_task, task_results, task_configs = {}, {}, {}
    task_versions, higher_is_better, n_samples, task_hashes = {}, {}, {}, {}
    for name in task_names:
        task = get_task(name, benchmark_path=args.benchmark_path, limit=args.limit)
        samples = task.run(model, gen_kwargs)
        samples_by_task[name] = samples
        # MT-Bench emits 2 records per question; n-samples counts records, like lm-eval
        # counts requests' docs
        task_results[name] = task.results_entry(len(samples))
        task_configs[name] = task.task_config()
        task_versions[name] = task.VERSION
        higher_is_better[name] = task.higher_is_better()
        n_samples[name] = {"original": len(samples), "effective": len(samples)}
        task_hashes[name] = compute_task_hash(samples)
        print(f"[mt_eval] {name}: {len(samples)} sample records", flush=True)

    config_block = {
        "model": args.model,
        "model_args": model_args,
        "batch_size": str(args.batch_size),
        "batch_sizes": [],
        "device": args.device,
        "use_cache": None,
        "limit": args.limit,
        "bootstrap_iters": 100000,
        "gen_kwargs": gen_kwargs,
        "random_seed": args.seed,
        "numpy_seed": args.seed,
        "torch_seed": args.seed,
        "fewshot_seed": args.seed,
    }
    results = build_results_dict(
        task_results=task_results,
        system_instruction=args.system_instruction,
        task_configs=task_configs,
        task_versions=task_versions,
        higher_is_better=higher_is_better,
        n_samples=n_samples,
        config_block=config_block,
        model_info=model.model_info(),
        task_hashes=task_hashes,
        start_unix=start_unix,
        total_seconds=time.perf_counter() - start,
    )

    writer = OutputWriter(args.output_path, model_args["pretrained"])
    results_file = writer.write(results, samples_by_task)
    print(f"[mt_eval] wrote {results_file}", flush=True)
    for name in task_names:
        print(
            f"[mt_eval] wrote {writer.dir / f'samples_{name}_{writer.date_id}.jsonl'}",
            flush=True,
        )


if __name__ == "__main__":
    main()
