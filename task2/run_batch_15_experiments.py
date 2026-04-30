import argparse
import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class ExperimentSpec:
    name: str
    model_type: str
    dnn_variant: str
    feature_type: str
    use_delta: bool
    use_delta_delta: bool


def _bool_str(v: bool) -> str:
    return "true" if v else "false"


def _load_last_record(jsonl_path: Path) -> Dict:
    if not jsonl_path.exists():
        return {}
    lines = [ln for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if not lines:
        return {}
    try:
        return json.loads(lines[-1])
    except Exception:
        return {}


def build_experiments() -> List[ExperimentSpec]:
    model_specs = [
        ("dnn_mlp", "dnn", "mlp"),
        ("dnn_bn_dropout", "dnn", "bn_dropout"),
        ("gmm", "gmm", "mlp"),
    ]
    feat_specs = [
        ("fbank", "fbank", False, False),
        ("mfcc", "mfcc", False, False),
        ("fbank_delta", "fbank", True, False),
        ("mfcc_delta", "mfcc", True, False),
        ("fbank_delta_deltadelta", "fbank", True, True),
    ]
    exps: List[ExperimentSpec] = []
    for model_tag, model_type, dnn_variant in model_specs:
        for feat_tag, feature_type, use_delta, use_delta_delta in feat_specs:
            exps.append(
                ExperimentSpec(
                    name=f"{model_tag}__{feat_tag}",
                    model_type=model_type,
                    dnn_variant=dnn_variant,
                    feature_type=feature_type,
                    use_delta=use_delta,
                    use_delta_delta=use_delta_delta,
                )
            )
    return exps


def run_batch(project_root: Path, python_bin: Path, gpu_ids: List[int], result_root: Path) -> None:
    run_dev = project_root / "task2" / "run_dev.py"
    if not run_dev.exists():
        raise FileNotFoundError(f"run_dev.py not found: {run_dev}")

    result_root.mkdir(parents=True, exist_ok=True)
    experiments = build_experiments()

    if not gpu_ids:
        raise ValueError("gpu_ids is empty")
    free_gpus = list(gpu_ids)
    pending = experiments.copy()
    running = []
    finished_rows: List[Dict] = []

    while pending or running:
        while pending and free_gpus:
            gpu_id = free_gpus.pop(0)
            exp = pending.pop(0)
            exp_dir = result_root / exp.name
            exp_dir.mkdir(parents=True, exist_ok=True)

            env = os.environ.copy()
            env.update(
                {
                    "CUDA_VISIBLE_DEVICES": str(gpu_id),
                    "PYTHONUNBUFFERED": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "TASK2_OUTPUT_DIR": str(exp_dir),
                    "TASK2_MODEL_TYPE": exp.model_type,
                    "TASK2_DNN_VARIANT": exp.dnn_variant,
                    "TASK2_FEATURE_TYPE": exp.feature_type,
                    "TASK2_USE_DELTA": _bool_str(exp.use_delta),
                    "TASK2_USE_DELTA_DELTA": _bool_str(exp.use_delta_delta),
                    "TASK2_USE_CMVN": "true",
                    "TASK2_CONTEXT_SIZE": "1",
                    "TASK2_FEATURE_DIM": "40",
                }
            )

            config_snapshot = {
                "model_type": exp.model_type,
                "dnn_variant": exp.dnn_variant,
                "feature_type": exp.feature_type,
                "use_delta": exp.use_delta,
                "use_delta_delta": exp.use_delta_delta,
                "use_cmvn": True,
                "context_size": 1,
                "feature_dim": 40,
                "assigned_gpu": gpu_id,
            }
            (exp_dir / "config_overrides.json").write_text(
                json.dumps(config_snapshot, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            run_log_path = exp_dir / "run_stdout.log"
            run_log = run_log_path.open("w", encoding="utf-8")
            cmd = [
                str(python_bin),
                str(run_dev),
                "--project-root",
                str(project_root),
                "--exp-name",
                exp.name,
                "--log-dir",
                str(exp_dir),
            ]
            proc = subprocess.Popen(
                cmd,
                cwd=project_root,
                stdout=run_log,
                stderr=subprocess.STDOUT,
                env=env,
            )
            running.append(
                {
                    "proc": proc,
                    "gpu_id": gpu_id,
                    "exp": exp,
                    "exp_dir": exp_dir,
                    "run_log": run_log,
                    "start_ts": time.time(),
                }
            )
            print(f"[launch] gpu={gpu_id} exp={exp.name}")

        if not running:
            continue

        time.sleep(5)
        next_running = []
        for item in running:
            proc = item["proc"]
            ret = proc.poll()
            if ret is None:
                next_running.append(item)
                continue

            item["run_log"].close()
            free_gpus.append(item["gpu_id"])
            exp = item["exp"]
            exp_dir = item["exp_dir"]
            record = _load_last_record(exp_dir / "task2_dev.jsonl")
            row = {
                "exp_name": exp.name,
                "return_code": ret,
                "status": record.get("status", "unknown"),
                "duration_sec": record.get("duration_sec", time.time() - item["start_ts"]),
                "acc": (record.get("result") or {}).get("acc"),
                "auc": (record.get("result") or {}).get("auc"),
                "eer": (record.get("result") or {}).get("eer"),
                "best_threshold": (record.get("result") or {}).get("best_threshold"),
                "exp_dir": str(exp_dir),
            }
            finished_rows.append(row)
            print(
                f"[done] exp={exp.name} rc={ret} status={row['status']} "
                f"auc={row['auc']} eer={row['eer']}"
            )
        running = next_running

    finished_rows.sort(key=lambda x: x["exp_name"])
    summary_json = result_root / "summary.json"
    summary_csv = result_root / "summary.csv"
    summary_json.write_text(json.dumps(finished_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "exp_name",
                "return_code",
                "status",
                "duration_sec",
                "acc",
                "auc",
                "eer",
                "best_threshold",
                "exp_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(finished_rows)

    print(f"[summary] total={len(finished_rows)}")
    print(f"[summary] json={summary_json}")
    print(f"[summary] csv={summary_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 15 task2 experiments in parallel")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path",
    )
    parser.add_argument(
        "--python-bin",
        type=Path,
        default=Path("/home/lihaochen/.conda/envs/vad/bin/python"),
        help="Python interpreter path (vad env)",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Deprecated: if set, use GPU ids 0..num_gpus-1",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help='Comma-separated physical GPU ids, e.g. "4,5,6,7"',
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=None,
        help="Result root directory; default is <project_root>/result/task2_batch_<timestamp>",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = time.strftime("%Y%m%d_%H%M%S")
    result_root = args.result_root
    if result_root is None:
        result_root = args.project_root / "result" / f"task2_batch_{ts}"
    if args.num_gpus is not None:
        gpu_ids = list(range(args.num_gpus))
    else:
        gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    run_batch(args.project_root.resolve(), args.python_bin.resolve(), gpu_ids, result_root.resolve())


if __name__ == "__main__":
    main()
