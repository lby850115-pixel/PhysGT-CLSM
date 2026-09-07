"""Run the bundled example and verify the frozen P3 inference path."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tifffile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "reference_results" / "example_p3.json").read_text())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    source = ROOT / REFERENCE["input"]
    assert sha256_file(source) == REFERENCE["input_sha256"], "Example input checksum mismatch"

    with tempfile.TemporaryDirectory(prefix="physgt_smoke_") as tmp:
        tmp_path = Path(tmp)
        pred = tmp_path / "pred"
        fig = tmp_path / "fig"
        result = tmp_path / "result"
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "PhysGT_CLSM.py"),
                "--img_dir",
                str(source.parent),
                "--out_pred",
                str(pred),
                "--out_fig",
                str(fig),
                "--out_res",
                str(result),
            ],
            check=True,
        )

        labels = tifffile.imread(pred / f"{source.stem}.tif")
        foreground_hash = hashlib.sha256(
            (labels > 0).astype(np.uint8).tobytes()
        ).hexdigest()
        with (result / "physegt_clsm_stats.csv").open(newline="") as stream:
            row = next(csv.DictReader(stream))

        assert int(labels.max()) == REFERENCE["n_instances"]
        assert int(row["area_median"]) == REFERENCE["median_area_px"]
        assert int(row["area_mean"]) == REFERENCE["mean_area_px"]
        assert foreground_hash == REFERENCE["foreground_sha256"]

    print("PASS: bundled example matches the frozen P3 reference.")


if __name__ == "__main__":
    main()
