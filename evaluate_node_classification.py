"""Evaluate saved DyG-WM / baseline checkpoints for dynamic node classification.

Example:
    python evaluate_node_classification.py --dataset_name dblp --model_name DyG-WM \
        --train_ratio 0.4 --gpu 0

Rebuilds each checkpoint written by ``train_node_classification.py`` on the
same probe split and re-runs the final frozen-probe (or supervised) test
evaluation.
"""

from __future__ import annotations

import json

import torch

from train_node_classification import (
    TEMPORAL_SSL_MODELS,
    aggregate_seed_runs,
    build_node_model,
    checkpoint_path,
    final_node_evaluation,
    load_run_config,
    parse_args,
    prepare_node_data,
)
from utils.utils import (
    create_logger,
    get_device,
    release_device_memory,
    set_random_seed,
)


def main() -> None:
    args = parse_args("Evaluate saved DyG-WM / baseline node-classification checkpoints")
    key, config = load_run_config(args)
    training = dict(config["training"])
    settings = dict(config.get(key, {}))
    seeds = args.seeds if args.seeds is not None else list(config.get("seeds", [42]))
    device = get_device(args.gpu)
    tag = f"{args.model_name}_ratio{args.train_ratio:g}"

    runs: dict[str, dict] = {}
    for seed in seeds:
        run_name = f"{tag}_seed{seed}"
        logger = create_logger(
            f"eval_{run_name}",
            args.log_dir / args.model_name / args.dataset_name / f"evaluate_{run_name}.log",
        )
        path = checkpoint_path(args, seed)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found; run train_node_classification.py with the same "
                "--dataset_name/--model_name/--train_ratio first"
            )
        checkpoint = torch.load(path, map_location="cpu")
        set_random_seed(seed)
        graph, _, probe_split = prepare_node_data(config, seed, device)
        torch.manual_seed(seed)
        model = build_node_model(key, config, graph, seed, device)
        if key in TEMPORAL_SSL_MODELS:
            model.prepare(graph)
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"loaded {path} (best epoch {checkpoint['best_epoch']})")
        result = final_node_evaluation(
            key, model, graph, probe_split, training, settings, seed, checkpoint["best_view"]
        )
        result["best_epoch"] = float(checkpoint["best_epoch"])
        result.update(checkpoint.get("extra", {}))
        logger.info(json.dumps({"final": result}))
        runs[str(seed)] = result
        del model, graph
        release_device_memory(device)

    summary = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "train_ratio": args.train_ratio,
        "seeds": seeds,
        "runs": runs,
        "aggregate": aggregate_seed_runs(runs),
    }
    summary_file = (
        args.save_result_dir / args.model_name / args.dataset_name / f"evaluate_{tag}.json"
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2))
    macro, micro = summary["aggregate"]["macro_f1"], summary["aggregate"]["micro_f1"]
    print(
        f"{args.model_name} on {args.dataset_name} (train ratio {args.train_ratio:g}) over "
        f"{len(seeds)} seed(s): Macro-F1 {macro['mean']:.4f} +- {macro['std']:.4f}, "
        f"Micro-F1 {micro['mean']:.4f} +- {micro['std']:.4f}"
    )


if __name__ == "__main__":
    main()
