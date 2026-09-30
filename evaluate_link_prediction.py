"""Evaluate saved DyG-WM / baseline checkpoints for dynamic link prediction.

Example:
    python evaluate_link_prediction.py --dataset_name wikipedia --model_name DyG-WM \
        --negative_sample_strategy historical --gpu 0

The checkpoints written by ``train_link_prediction.py`` are rebuilt on the
same data split and scored under the requested evaluation negatives
(random / historical / inductive) and setting (transductive / inductive).
"""

from __future__ import annotations

import json

import torch

from train_link_prediction import (
    aggregate_seed_runs,
    assert_full_event_coverage,
    build_model,
    final_evaluation,
    parse_args,
    prepare_link_data,
    result_suffix,
)
from utils.load_configs import load_config, model_key, training_arguments
from utils.negative_sampling import normalize_negative_strategy
from utils.inductive_setting import normalize_setting
from utils.utils import (
    create_logger,
    device_description,
    get_device,
    release_device_memory,
    set_random_seed,
)


def main() -> None:
    args = parse_args("Evaluate saved DyG-WM / baseline checkpoints")
    key = model_key(args.model_name)
    negative_strategy = normalize_negative_strategy(args.negative_sample_strategy)
    setting = normalize_setting(args.setting)
    config = load_config(args.config, args.dataset_name)
    training = training_arguments(config, key)
    seeds = args.seeds if args.seeds is not None else list(config.get("seeds", [0]))
    device = get_device(args.gpu)
    suffix = result_suffix(negative_strategy, setting)
    model_dir = args.model_name

    runs: dict[str, dict] = {}
    for seed in seeds:
        run_name = f"{model_dir}_seed{seed}{suffix}"
        logger = create_logger(
            f"eval_{run_name}",
            args.log_dir / model_dir / args.dataset_name / f"evaluate_{run_name}.log",
        )
        set_random_seed(seed)
        data = prepare_link_data(config, device, negative_strategy, setting, logger)
        if key != "edgebank":
            torch.manual_seed(seed)
        model = build_model(key, config, data, device)
        best_epoch = 0
        if key != "edgebank":
            checkpoint_path = (
                args.save_model_dir / model_dir / args.dataset_name / f"{model_dir}_seed{seed}"
                f"{'_inductive_setting' if setting == 'inductive' else ''}.pkl"
            )
            if not checkpoint_path.exists():
                raise FileNotFoundError(
                    f"{checkpoint_path} not found; run train_link_prediction.py with the "
                    "same --dataset_name/--model_name/--setting first"
                )
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["state_dict"])
            best_epoch = int(checkpoint.get("best_epoch", 0))
            logger.info(f"loaded {checkpoint_path} (best epoch {best_epoch})")
        logger.info(
            json.dumps(
                {
                    "dataset": args.dataset_name,
                    "model": args.model_name,
                    "seed": seed,
                    "negative_sample_strategy": negative_strategy,
                    "setting": setting,
                    "device": device_description(device),
                }
            )
        )
        set_random_seed(seed)
        validation, test = final_evaluation(model, data, training)
        test["best_epoch"] = float(best_epoch)
        result = {"validation": validation, "test": test}
        assert_full_event_coverage(
            args.model_name,
            result,
            data.split if data.inductive is None else data.inductive.final_split(data.split),
            data.link_cfg,
        )
        logger.info(json.dumps({"final": result}))
        runs[str(seed)] = result
        del model, data
        release_device_memory(device)

    summary = {
        "dataset": args.dataset_name,
        "model": args.model_name,
        "negative_sample_strategy": negative_strategy,
        "setting": setting,
        "seeds": seeds,
        "runs": runs,
        "aggregate": aggregate_seed_runs(runs),
    }
    summary_file = (
        args.save_result_dir
        / model_dir
        / args.dataset_name
        / f"evaluate_{model_dir}{suffix}.json"
    )
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, indent=2))
    test_summary = summary["aggregate"]["test"]
    print(
        f"{args.model_name} on {args.dataset_name} ({negative_strategy}, {setting}) "
        f"over {len(seeds)} seed(s): test AP {test_summary['ap']['mean']:.4f} "
        f"+- {test_summary['ap']['std']:.4f}, AUC {test_summary['auc']['mean']:.4f} "
        f"+- {test_summary['auc']['std']:.4f}"
    )


if __name__ == "__main__":
    main()
