import argparse
import csv
import datetime
import math
import re
import subprocess

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def detect_device_name():
    try:
        result = subprocess.run(
            ["vulkaninfo"], capture_output=True, text=True, timeout=10
        )
        m = re.search(r"deviceName\s*=\s*(.+)", result.stdout)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return None


def load(paths):
    # data: { backend: { sort: { n: { gpu, cpu } } } }
    # versions: { backend: version_string }
    data = {}
    versions = {}
    for path in paths:
        file_meta = {}
        file_backends = set()
        with open(path, newline="") as f:
            reader = csv.DictReader(
                (line for line in f
                 if not line.startswith("#") or _parse_meta(line, file_meta)),
            )
            for row in reader:
                backend = row["backend"]
                file_backends.add(backend)
                sort = row["sort"]
                n = int(row["n"])
                data.setdefault(backend, {}).setdefault(sort, {})[n] = {
                    "gpu": float(row["gpu_gitems_s"]),
                    "cpu": float(row["cpu_gitems_s"]),
                }
        if "version" in file_meta:
            for backend in file_backends:
                versions[backend] = file_meta["version"]
    return data, versions


def _parse_meta(line, meta):
    m = re.match(r"#\s*(\w+):\s*(.+)", line)
    if m:
        meta[m.group(1)] = m.group(2).strip()
    return False  # always filter out comment lines from CSV rows


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from one or more CSV files."
    )
    parser.add_argument("csvs", nargs="+", metavar="results.csv")
    parser.add_argument("--output", default="results.png", metavar="FILE")
    args = parser.parse_args()

    device = detect_device_name()

    data, versions = load(args.csvs)

    backends = sorted(data.keys())
    ns = sorted(next(iter(next(iter(data.values())).values())).keys())

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    backend_color = {b: colors[i % len(colors)]
                     for i, b in enumerate(backends)}

    fig, (ax_keys, ax_kv) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    meta_parts = []
    if device:
        meta_parts.append(f"Device: {device}")
    backend_labels = {"vulkan": "VRDX", "cuda": "CUB"}
    for backend in backends:
        if backend in versions:
            label = backend_labels.get(backend, backend)
            meta_parts.append(f"{label} {versions[backend]}")
    meta_parts.append(datetime.date.today().strftime("%Y%m%d"))
    if meta_parts:
        fig.suptitle("  |  ".join(meta_parts), fontsize=10, color="gray")

    def plot_panel(ax, sort, title):
        for backend in backends:
            color = backend_color[backend]
            for timing, ls in [("gpu", "-"), ("cpu", "--")]:
                ys = [data[backend][sort][n][timing] for n in ns]
                ax.plot(ns, ys, color=color, linestyle=ls,
                        label=f"{backend} {timing.upper()}")
        ax.set_ylabel("Throughput (GItems/s)")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
        )
        for n in ns:
            if n > 0 and (n & (n - 1)) == 0:
                ax.text(n, 0.01, f"$2^{{{int(math.log2(n))}}}$",
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=8, color="gray")
        ax.xaxis.set_major_locator(ticker.FixedLocator(ns[3::4]))
        ax.set_xticks(ns, minor=True)
        ax.yaxis.set_major_locator(ticker.AutoLocator())
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))
        ax.legend()
        ax.grid(True, which="major", linestyle=":", alpha=0.5)
        ax.grid(True, which="minor", linestyle=":", alpha=0.2)

    plot_panel(ax_keys, "keys", "Keys-Only Sort")
    plot_panel(ax_kv,   "kv",   "Key-Value Sort")

    ax_kv.set_xlabel("N (elements)")
    plt.setp(ax_kv.get_xticklabels(), rotation=90)
    plt.tight_layout()

    plt.savefig(args.output, dpi=150)
    print(f"Saved {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
