import argparse
import json
from pathlib import Path

from scripts.data_quality_audit import run_audit


def calibration_report_schema() -> dict:
    return {
        "tiny_overfit": {"status": "pending", "models": []},
        "neg_mode_sensitivity": {"status": "pending", "modes": ["random", "popularity", "mixed"]},
        "seed_stability": {"status": "pending", "seeds": []},
    }


def build_report(data_dir: Path) -> dict:
    report = calibration_report_schema()
    report["data_quality"] = run_audit(data_dir)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate a calibration report scaffold.")
    parser.add_argument("--data_dir", required=True, type=Path, help="Directory containing processed dataset CSV/JSON files.")
    parser.add_argument("--out", required=True, type=Path, help="Path to write the calibration report JSON.")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    report = build_report(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
