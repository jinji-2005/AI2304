import argparse
import sys
import time
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import FeatureConfig, FrameConfig, ModelConfig
from experiment_logger import write_experiment_log
from pipeline import run_test_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task2 test inference entrypoint")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "test_label.txt",
        help="Output label file path",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="",
        help="Optional experiment name for logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    status = "success"
    err_msg = None
    try:
        run_test_pipeline(args.project_root, args.output)
        print(f"Task2 test labels written to: {args.output}")
    except Exception as exc:
        status = "failed"
        err_msg = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        log_path = write_experiment_log(
            project_root=args.project_root,
            task="task2",
            mode="test",
            command=sys.argv,
            status=status,
            duration_sec=time.time() - t0,
            config={
                "frame": asdict(FrameConfig()),
                "feature": asdict(FeatureConfig()),
                "model": asdict(ModelConfig()),
            },
            result={"output_path": str(args.output)},
            extra={"exp_name": args.exp_name},
            error=err_msg,
        )
        print(f"Experiment log appended to: {log_path}")


if __name__ == "__main__":
    main()
