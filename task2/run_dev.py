import argparse
import sys
import time
import os
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FeatureConfig, FrameConfig, ModelConfig
from experiment_logger import write_experiment_log
from pipeline import run_dev_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2 dev evaluation entrypoint")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help="Optional experiment name for logging",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=None,
        help="Optional experiment log directory override",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    result = None
    status = "success"
    err_msg = None
    try:
        result = run_dev_pipeline(args.project_root)
        print(result)
    except Exception as exc:
        status = "failed"
        err_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        env_log_dir = os.getenv("TASK2_OUTPUT_DIR", "").strip()
        effective_log_dir = args.log_dir
        if effective_log_dir is None and env_log_dir:
            effective_log_dir = Path(env_log_dir)
        log_path = write_experiment_log(
            project_root=args.project_root,
            task="task2",
            mode="dev",
            command=sys.argv,
            status=status,
            duration_sec=time.time() - t0,
            config={
                "frame": asdict(FrameConfig()),
                "feature": asdict(FeatureConfig()),
                "model": asdict(ModelConfig()),
            },
            result=result if isinstance(result, dict) else {},
            extra={"exp_name": args.exp_name},
            error=err_msg,
            log_dir=effective_log_dir,
        )
        print(f"Experiment log appended to: {log_path}")


if __name__ == "__main__":
    main()
