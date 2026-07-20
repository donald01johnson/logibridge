#!/usr/bin/env python3
"""Generate the annotated LogiEdge Pareto chart from final F2 results."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

VARIANTS = ["M1 FP32", "M2 INT8", "M3 Structured INT8"]
SIZE_KB = np.array([5.477, 4.781, 4.219])
MEAN_MS = np.array([0.003501, 0.013597, 0.013923])
P95_MS = np.array([0.004930, 0.015308, 0.014422])
ENERGY_MJ = np.array([0.051994, 0.203961, 0.208850])
ACCURACY = np.array([100.0, 100.0, 100.0])
CRITICAL_RECALL = np.array([100.0, 100.0, 100.0])


def main():
    output_dir = Path("optimisation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    frontier_order = np.argsort(SIZE_KB)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6.5), dpi=180)

    ax1.plot(
        SIZE_KB[frontier_order], MEAN_MS[frontier_order], "--",
        linewidth=1.5, label="Pareto frontier"
    )
    ax1.scatter(SIZE_KB, MEAN_MS, s=90, zorder=3)
    offsets = {
        "M1 FP32": (8, 10),
        "M2 INT8": (8, -24),
        "M3 Structured INT8": (-118, 10),
    }
    for name, x, y, p95, acc, recall in zip(
        VARIANTS, SIZE_KB, MEAN_MS, P95_MS, ACCURACY, CRITICAL_RECALL
    ):
        ax1.annotate(
            f"{name}\nMean {y:.6f} ms | p95 {p95:.6f} ms\n"
            f"Accuracy {acc:.0f}% | Critical recall {recall:.0f}%",
            xy=(x, y), xytext=offsets[name], textcoords="offset points",
            fontsize=8.5, arrowprops=dict(arrowstyle="->", linewidth=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )
    ax1.annotate(
        "Recommended: M1\nFastest and lowest estimated energy;\n"
        "size difference is immaterial on Raspberry Pi 5",
        xy=(SIZE_KB[0], MEAN_MS[0]), xytext=(-205, 72),
        textcoords="offset points", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="->", linewidth=1.2),
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.95),
    )
    ax1.set_title("A. Pareto Trade-off: Model Size vs Mean Latency")
    ax1.set_xlabel("Model file size (KB, lower is better)")
    ax1.set_ylabel("Mean inference latency (ms, lower is better)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="center right")
    ax1.text(0.02, 0.02, "Preferred direction: lower-left",
             transform=ax1.transAxes, fontsize=8.5)

    ax2.plot(
        SIZE_KB[frontier_order], ENERGY_MJ[frontier_order], "--",
        linewidth=1.5, label="Trade-off path"
    )
    ax2.scatter(SIZE_KB, ENERGY_MJ, s=90, zorder=3)
    energy_offsets = {
        "M1 FP32": (8, 10),
        "M2 INT8": (8, -22),
        "M3 Structured INT8": (-118, 10),
    }
    for name, x, energy in zip(VARIANTS, SIZE_KB, ENERGY_MJ):
        ax2.annotate(
            f"{name}\n{energy:.6f} mJ", xy=(x, energy),
            xytext=energy_offsets[name], textcoords="offset points",
            fontsize=8.5, arrowprops=dict(arrowstyle="->", linewidth=0.8),
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
        )
    ax2.set_title("B. Supporting Trade-off: Model Size vs Estimated Energy")
    ax2.set_xlabel("Model file size (KB, lower is better)")
    ax2.set_ylabel("Estimated energy per inference (mJ, lower is better)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="center right")
    ax2.text(
        0.02, 0.02,
        "Energy uses psutil CPU% and a 15 W laptop TDP estimate",
        transform=ax2.transAxes, fontsize=8.5,
    )

    fig.suptitle(
        "LogiEdge Final Model-Variant Pareto Analysis\n"
        "All variants achieved 100% validation accuracy and 100% Critical recall",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_dir / "pareto_chart.png", bbox_inches="tight")
    fig.savefig(output_dir / "pareto_chart.pdf", bbox_inches="tight")
    plt.close(fig)
    print(output_dir / "pareto_chart.png")
    print(output_dir / "pareto_chart.pdf")


if __name__ == "__main__":
    main()
