import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=Path, default="results")
    parser.add_argument("--num", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    target_dir = Path(args.target_dir)
    dir_list = sorted(target_dir.glob("*"))

    dir_list = dir_list[-args.num :]

    for d in dir_list:
        csv_file = d / "result.csv"
        if not csv_file.exists():
            print(f"Skip {d.name}")
            continue
        df = pd.read_csv(csv_file)
        max_score = df["reward_mean"].max()
        print(f"Max score in {d.name}: {max_score}")
