"""
run_pipeline.py

Sequential runner for the STAT 636 network/epidemic project.

HOW TO USE
1. Put this file in the root of your stat_project folder.
2. Edit PIPELINE below so the filenames match the scripts in your project.
   - Add your empirical-network preparation script before run_null_models.py.
   - Add your empirical epidemic / Chunk 8C generation script(s) in the
     appropriate place.
   - Add the exact Chunk 9, 10, 11, and 12 script filenames at the end.
3. Run from PyCharm, or from a terminal:
       python run_pipeline.py
   Preview the order without executing:
       python run_pipeline.py --dry-run

The runner stops immediately if a script fails. It does not run config.py or
vaccination.py because those are normally imported modules, not pipeline
entry points. Review the order before running this on important results.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# This file is expected to live in the project root.
PROJECT_DIR = Path(__file__).resolve().parent
LOG_DIR = PROJECT_DIR / "logs"

# EDIT THIS LIST to match the exact executable script filenames in your folder.
# Keep upstream data/network generation before simulations, and simulations
# before the Chunk 9-12 analysis/report scripts.
PIPELINE = [
    "config.py",
    "data_loader.py",
    "network_analysis.py",
    "spectral_analysis.py",
    "vaccination.py",
    "sir_simulation.py",
    "visualization.py",
    "run_null_models.py",
    "run_null_model_epidemics.py",
    "visualize_null_model_epidemics.py",
    "advanced_statistical_analysis.py",
    "final_results_integration.py",
    "chunk11_final_interpretation.py",
    "chunk12_final_research_report.py",
]

def main() -> int:
    parser = argparse.ArgumentParser(description="Run the STAT 636 project pipeline.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the configured execution order without running scripts.",
    )
    parser.add_argument(
        "--start-at",
        help="Resume at a script filename in PIPELINE (exact match).",
    )
    args = parser.parse_args()

    if not PIPELINE:
        print("PIPELINE is empty. Edit PIPELINE in run_pipeline.py first.")
        return 2

    scripts = PIPELINE[:]
    if args.start_at:
        if args.start_at not in scripts:
            print(f"--start-at script not found in PIPELINE: {args.start_at}")
            return 2
        scripts = scripts[scripts.index(args.start_at):]

    print(f"Project folder: {PROJECT_DIR}")
    print("\nExecution order:")
    for i, name in enumerate(scripts, start=1):
        print(f"  {i}. {name}")

    if args.dry_run:
        missing = [name for name in scripts if not (PROJECT_DIR / name).is_file()]
        if missing:
            print("\nFiles not found (edit PIPELINE or check filenames):")
            for name in missing:
                print(f"  - {name}")
            return 1
        print("\nDry run complete. No scripts were executed.")
        return 0

    missing = [name for name in scripts if not (PROJECT_DIR / name).is_file()]
    if missing:
        print("\nCannot start: these configured scripts were not found:")
        for name in missing:
            print(f"  - {name}")
        print("Edit PIPELINE in run_pipeline.py to use your exact filenames.")
        return 1

    LOG_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"pipeline_{timestamp}.log"

    print(f"\nLog file: {log_path}")
    with log_path.open("w", encoding="utf-8") as log:
        for index, name in enumerate(scripts, start=1):
            script_path = PROJECT_DIR / name
            command = [sys.executable, str(script_path)]

            heading = f"\n[{index}/{len(scripts)}] Running {name}\n"
            print(heading, end="")
            log.write(heading)
            log.flush()

            try:
                completed = subprocess.run(
                    command,
                    cwd=PROJECT_DIR,
                    text=True,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            except OSError as exc:
                message = f"Could not start {name}: {exc}"
                print(message)
                log.write(message + "\n")
                return 1

            if completed.returncode != 0:
                message = (
                    f"\nSTOPPED: {name} exited with code "
                    f"{completed.returncode}. See {log_path}"
                )
                print(message)
                log.write(message + "\n")
                return completed.returncode or 1

            print(f"Completed: {name}")

    print(f"\nPipeline completed successfully. Log saved to: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
