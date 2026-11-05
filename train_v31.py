"""Train original V3.1's 18 configurations with the shared normal-only pipeline."""
import argparse
import json
from pathlib import Path

import pandas as pd

from train_v3 import train, sha


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=root / "config/V31_TRAINING_CONFIG.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("version") != "V3.1" or len(config["configurations"]) != 18:
        raise ValueError("Expected the frozen V3.1 configuration family")
    data = pd.read_csv(args.features, float_precision="round_trip")
    index = train(data, config, args.output)
    (args.output / "INPUT_HASHES.json").write_text(json.dumps(dict(
        features_sha256=sha(args.features), config_sha256=sha(args.config),
        training_entrypoint_sha256=sha(__file__)), indent=2), encoding="utf-8")
    print(f"Saved {len(index)} V3.1 models to {args.output}; V3 baseline is not replaced")


if __name__ == "__main__":
    main()
