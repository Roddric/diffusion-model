"""Plot paired origin-level losses from the post-hoc robustness audit."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


MODEL_NAME = "Phase2F-Validation-Pooled-Diffusion"


def run(args):
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    locked = json.loads(Path(args.locked_result).read_text(encoding="utf-8"))
    dates = pd.to_datetime([row["target_end"] for row in locked["origins"]])
    comparisons = audit["dependence_robust_primary_comparisons"]

    metrics = (
        ("state_energy_score", "State energy loss difference"),
        ("state_rmse", "State RMSE difference"),
    )
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = {"Gaussian-VAR": "#1768ac", "Student-t-VAR": "#cc6b32"}
    for axis, (metric, label) in zip(axes, metrics):
        for baseline in ("Gaussian-VAR", "Student-t-VAR"):
            values = np.asarray(
                comparisons[baseline][metric]["origin_loss_differences"]
            )
            axis.plot(
                dates,
                values,
                marker="o",
                markersize=3.5,
                linewidth=1.1,
                color=colors[baseline],
                alpha=0.85,
                label=f"Pool minus {baseline}",
            )
            axis.axhline(
                values.mean(),
                color=colors[baseline],
                linestyle="--",
                linewidth=1.1,
                alpha=0.9,
            )
        axis.axhline(0.0, color="black", linewidth=0.9)
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
    axes[0].legend(loc="best", frameon=False, ncol=2)
    axes[-1].set_xlabel("20-day target-window end")
    fig.suptitle(
        "Retrospective holdout: paired origin losses\n"
        "Negative values favor the VAR–diffusion pool",
        fontsize=12,
    )
    fig.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audit",
        default=(
            "research_output/sp500_confirmation/"
            "posthoc_dependence_robustness.json"
        ),
    )
    parser.add_argument(
        "--locked-result",
        default="research_output/sp500_confirmation/confirmation.json",
    )
    parser.add_argument(
        "--output",
        default=(
            "research_output/sp500_confirmation/"
            "paired_origin_robustness.png"
        ),
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
