import csv
import math
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load(paths):
    # { backend: { sort: { n: { gpu, cpu } } } }
    data = {}
    for path in paths:
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                backend = row["backend"]
                sort = row["sort"]
                n = int(row["n"])
                data.setdefault(backend, {}).setdefault(sort, {})[n] = {
                    "gpu": float(row["gpu_gitems_s"]),
                    "cpu": float(row["cpu_gitems_s"]),
                }
    return data


def main():
    if len(sys.argv) < 2:
        print("Usage: plot.py <results.csv> [more.csv ...] [output.png]")
        sys.exit(1)

    # last arg is output path if it ends with .png, otherwise default
    args = sys.argv[1:]
    if args[-1].endswith(".png"):
        png_path, csv_paths = args[-1], args[:-1]
    else:
        png_path, csv_paths = "results.png", args

    data = load(csv_paths)

    backends = sorted(data.keys())
    ns = sorted(next(iter(next(iter(data.values())).values())).keys())

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    backend_color = {b: colors[i % len(colors)]
                     for i, b in enumerate(backends)}

    _, (ax_keys, ax_kv) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    def plot_panel(ax, sort, title):
        for backend in backends:
            color = backend_color[backend]
            for timing, ls in [("gpu", "-"), ("cpu", "--")]:
                ys = [data[backend][sort][n][timing] for n in ns]
                ax.plot(ns, ys, color=color, linestyle=ls,
                        label=f"{backend} {timing.upper()}")
        ax.set_ylabel("Throughput (GItems/s)")
        ax.set_title(title)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
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

    plt.savefig(png_path, dpi=150)
    print(f"Saved {png_path}")
    plt.show()


if __name__ == "__main__":
    main()
