"""Redraw saved comparison values; no raw labels, model calls, or host paths."""
import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a fresh output directory")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = Path(__file__).resolve().parents[1]
    packets = sorted((root / "results/figure_packets").glob("*.json"))
    args.output.mkdir(parents=True)
    for path in packets:
        packet = json.loads(path.read_text())
        bars = packet["bars"]
        for bar in bars:
            if len(bar["values"]) != len(packet["cohort"]) or set(bar["assay_ids"]) != set(packet["cohort"]):
                raise ValueError("Incomplete plotted cohort: " + bar["name"])
            if not math.isclose(math.fsum(bar["values"]) / len(bar["values"]), bar["mean"], abs_tol=1e-12):
                raise ValueError("Saved mean differs from per-assay values: " + bar["name"])
        fig, ax = plt.subplots(figsize=(max(8, len(bars) * .8), 5.5), layout="constrained")
        colors = ["#496a81" if bar.get("kind") == "llm" else "#9caaa0" for bar in bars]
        values = [bar["mean"] for bar in bars]
        ax.bar(range(len(bars)), values, color=colors, width=.65)
        for index, bar in enumerate(bars):
            # Fixed deterministic offsets expose assay variation without random jitter.
            n = len(bar["values"])
            offsets = [0 if n == 1 else (i / (n-1) - .5) * .35 for i in range(n)]
            ax.scatter([index + offset for offset in offsets], bar["values"], s=8,
                       color="#172a35", alpha=.3, zorder=3)
        ax.set_xticks(range(len(bars)), [bar["name"] for bar in bars], rotation=35, ha="right")
        ax.set_ylabel("Signed Spearman" if packet["metric"] == "spearman" else "AUROC")
        ax.set_title(f"{packet['title']} · {len(packet['cohort'])} assays")
        ax.spines[["top", "right"]].set_visible(False)
        ax.axhline(0, color="#53616a", linewidth=.6)
        for ext in ("png", "svg", "pdf"):
            fig.savefig(args.output / f"{packet['system']}.{ext}", dpi=160)
        plt.close(fig)
    (args.output / "README.txt").write_text(
        "Portable redraw of saved numeric packets with Matplotlib defaults.\n"
        "See figures/published for byte-identical historical publication figures.\n"
        "Best-arm plots are descriptive same-cohort selection, not held-out model selection.\n")
    print(f"Validated and redrew {len(packets)} figures; model calls: 0")


if __name__ == "__main__":
    main()

