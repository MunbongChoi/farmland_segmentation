"""Create reproducible train/validation/test manifests without copying rasters."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .datasets.dataset import deterministic_partition, discover_pairs
from .utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="분할 manifest 생성")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/manifests")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = config["dataset"]
    train = discover_pairs(dataset["root_dir"], dataset["train"])
    validation_all = discover_pairs(dataset["root_dir"], dataset["validation"])
    validation, test = deterministic_partition(validation_all, float(dataset["validation_test_fraction"]), int(config["project"]["seed"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name, pairs in (("train", train), ("validation", validation), ("test", test)):
        with (output / f"{name}.csv").open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image", "mask"])
            writer.writerows((str(pair.image), str(pair.mask)) for pair in pairs)


if __name__ == "__main__":
    main()

