"""Show saved annotation coverage and the unattended runner's latest status. No inference."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from neurosym.annotation import annotation_passes
from neurosym.io import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/annotation.json")
    args = parser.parse_args()
    config = read_json(args.config)
    output = ROOT / config["output"]
    index = read_json(ROOT / config["corpus"] / "index.json")
    passes = annotation_passes(config)
    total = done = 0
    for pass_name in passes:
        for entry in index["stories"]:
            story = read_json(ROOT / config["corpus"] / entry["path"])
            count = sum((output / pass_name / story["story_id"] / (unit["id"] + ".json")).is_file()
                        for unit in story["units"])
            total += len(story["units"])
            done += count
            print(f"{pass_name}/{story['story_id']}: {count}/{len(story['units'])} saved")
    print(f"TOTAL: {done}/{total} saved")
    path = output / "progress.json"
    if path.is_file():
        progress = read_json(path)
        print(f"Last recorded status: {progress['status']} | updated {progress['updated_utc']}")
        current = progress.get("current", {})
        print("Current:", current)
        for failure in progress.get("unresolved", []):
            print(f"DEFERRED {failure['pass']}/{failure['unit_id']}: {failure['error']}")
    print("Run lock:", "present" if (output / "RUNNING.lock").exists() else "absent")
    print("Saved counts report coverage, not semantic accuracy. Progress timestamps can be stale if a process was killed.")


if __name__ == "__main__":
    main()
