import argparse
import json
from .pipeline import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a retail fraud classifier.")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("train")
    command.add_argument("--train-path", required=True)
    command.add_argument("--output-dir", default="artifacts")
    command.add_argument("--target")
    args = parser.parse_args()
    print(json.dumps(train(args.train_path, args.output_dir, args.target), indent=2))
