import argparse
from pathlib import Path
import librosa
from pipeline import run_dev_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task1 dev evaluation entrypoint")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_dev_pipeline(args.project_root)
    print(result)


if __name__ == "__main__":
    main()
