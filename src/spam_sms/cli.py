from __future__ import annotations

import argparse
import json

from . import pipeline


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Spam SMS classifier")
    sub = p.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train the classifier")
    train_p.add_argument("--test-size", type=float, default=0.2)
    train_p.add_argument("--seed", type=int, default=7)

    eval_p = sub.add_parser("evaluate", help="Evaluate the classifier")
    eval_p.add_argument("--test-size", type=float, default=0.2)
    eval_p.add_argument("--seed", type=int, default=7)

    pred_p = sub.add_parser("predict", help="Predict a single message")
    pred_p.add_argument("message", type=str)
    pred_p.add_argument("--threshold", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    if args.command == "train":
        metrics = pipeline.train(test_size=args.test_size, seed=args.seed)
        print(json.dumps(metrics, indent=2))
    elif args.command == "evaluate":
        metrics = pipeline.train(test_size=args.test_size, seed=args.seed)
        print(json.dumps(metrics, indent=2))
    elif args.command == "predict":
        pred = pipeline.predict(args.message, threshold=args.threshold)
        print(json.dumps(pred, indent=2))


if __name__ == "__main__":
    main()
