"""Build the V3 native-sample features from explicitly supplied saved sessions."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "original"))
import train_xarm_v3 as original
from v3.feature_helpers import describe, current_fall


class SharedSources(original.Sources):
    """Read original nested captures or the byte-identical shared flat CSV layout."""
    def track(self, path):
        path = Path(path)
        if not path.is_file() and path.parent.name in ("sensors", "robot"):
            flat = path.parent.parent / path.name
            if flat.is_file():
                path = flat
        return super().track(path)

    def json(self, path):
        path = Path(path)
        if path.name == "manifest.json" and not path.exists():
            cycles = self.csv(path.parent / "cycles.csv")
            ids = cycles.session_id.dropna().unique()
            if len(ids) != 1:
                raise ValueError("A shared session must contain one actual session_id")
            return {"session_id": ids[0]}
        return super().json(path)


def build(manifest, config, version="V3"):
    data = pd.read_csv(manifest, float_precision="round_trip")
    required = {"cycle_id", "session_id", "cohort", "source", "No3_begin_ns", "No3_end_ns"}
    if required - set(data):
        raise ValueError("Missing manifest fields: " + str(required - set(data)))
    if data.cycle_id.duplicated().any() or data[list(required)].isna().any().any():
        raise ValueError("Invalid or duplicate manifest cycles")
    if not (data.No3_begin_ns < data.No3_end_ns).all():
        raise ValueError("Cycle time bounds must increase")
    data["source"] = data.source.map(lambda p: str((manifest.parent / p).resolve()))
    source, rows = SharedSources(), []
    clock = "host_receive_monotonic_ns"
    for name, group in data.groupby("source", sort=False):
        folder = Path(name)
        raw = source.json(folder / "manifest.json")
        if not group.session_id.eq(raw["session_id"]).all():
            raise ValueError("Session identity mismatch")
        ina = source.csv(folder / "sensors/measurements.csv")
        adxl = source.csv(folder / "sensors/adxl345_1.csv")
        position = source.csv(folder / "robot/single_queries.csv")
        if not ina.valid.eq(1).all() or not ina.status.eq("OK").all():
            raise ValueError("INA quality failure")
        for record in group.to_dict("records"):
            begin, end = record["No3_begin_ns"], record["No3_end_ns"]
            pos = position[position.cycle_id.eq(record["cycle_id"]) & position.phase.eq("No3_A_to_B")]
            if not pos.status.eq("complete").all() or not pos.value_valid.eq(1).all():
                raise ValueError("Invalid position values; audit or exclude the cycle before feature building")
            values = original.full_features(original.subset(ina, clock, begin, end, True),
                original.subset(adxl, clock, begin, end, True),
                original.subset(pos, "response_ns", begin, end, True), describe)
            record.update(values)
            rows.append(record)
    data = pd.DataFrame(rows)
    data, wave, parity = original.extract(data, source, describe, current_fall, config["configurations"]["R1"])
    reference = data.loc[data.cohort.eq("train"), "id4_end_raw"].median()
    data[original.END] = (data.id4_end_raw - reference).abs()
    if version == "V3.1":
        import train_xarm_v31
        data, _, _, _ = train_xarm_v31.raw_extras(data, wave, original, source)
    assert all(original.sha(path) == digest for path, digest in source.hashes.items())
    return data, parity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--version", choices=["V3", "V3.1"], default="V3")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new feature output")
    config = json.loads((ROOT / "config/V3_TRAINING_CONFIG.json").read_text(encoding="utf-8"))
    frame, parity = build(args.manifest.resolve(), config, args.version)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, encoding="utf-8-sig")
    parity.to_csv(args.output.with_suffix(".parity.csv"), index=False, encoding="utf-8-sig")
    print(f"Built {len(frame)} cycles; hardware access none")


if __name__ == "__main__":
    main()
