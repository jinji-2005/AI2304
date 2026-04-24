import argparse
from pathlib import Path

from pipeline import run_test_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Task1 test inference entrypoint")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_test_pipeline(args.project_root, args.output)
    print(f"Task1 test labels written to: {args.output}")


if __name__ == "__main__":
    main()
