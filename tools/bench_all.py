#!/usr/bin/env python3
"""Run all available benchmark backends, collect CSVs, and generate a plot.

Usage:
    python tools/bench_all.py [options]
    python tools/bench_all.py --build-dir build/Release --no-verify
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BACKENDS = ["vulkan", "fuchsia", "cuda"]


def find_bench(build_dir: Path) -> Path:
    for candidate in [
        build_dir / "bench",
        build_dir / "bench.exe",
        build_dir / "Release" / "bench",
        build_dir / "Release" / "bench.exe",
    ]:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"bench executable not found under {build_dir}. Run cmake --build first."
    )


def run_backend(bench: Path, backend: str, csv: Path, extra: list[str]) -> bool:
    cmd = [str(bench), backend, "-o", str(csv)] + extra
    print(f"\n=== {backend} ===")
    print("$", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"[{backend}] exited {result.returncode} — skipping.")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run benchmark backends, save CSVs, and plot."
    )
    parser.add_argument(
        "--build-dir",
        default="build",
        metavar="DIR",
        help="CMake build directory (default: build)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help="Output directory (default: benchmarks/<timestamp>)",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=DEFAULT_BACKENDS,
        metavar="BACKEND",
        help=f"Backends to run (default: {' '.join(DEFAULT_BACKENDS)})",
    )
    parser.add_argument("--no-plot", action="store_true", help="Skip plot step")
    parser.add_argument(
        "--no-verify", action="store_true", help="Pass --no-verify to bench"
    )
    parser.add_argument(
        "--validation", action="store_true", help="Pass --validation to bench"
    )
    args = parser.parse_args()

    build_dir = Path(args.build_dir)
    if not build_dir.is_absolute():
        build_dir = REPO_ROOT / build_dir

    bench = find_bench(build_dir)
    print(f"bench: {bench}")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else REPO_ROOT / "benchmarks" / timestamp
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"output: {out_dir}")

    extra = []
    if args.no_verify:
        extra.append("--no-verify")
    if args.validation:
        extra.append("--validation")

    csvs = []
    for backend in args.backends:
        csv = out_dir / f"{backend}.csv"
        if run_backend(bench, backend, csv, extra):
            csvs.append(csv)

    if not csvs:
        print("\nAll backends failed.")
        sys.exit(1)

    if not args.no_plot:
        plot_out = out_dir / "results.png"
        cmd = (
            [sys.executable, str(REPO_ROOT / "tools" / "plot.py")]
            + [str(c) for c in csvs]
            + ["--output", str(plot_out)]
        )
        print(f"\n=== plot ===")
        print("$", " ".join(cmd))
        subprocess.run(cmd, check=True)

    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
