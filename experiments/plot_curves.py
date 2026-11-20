"""Training-curve figures from experiments/curves/curves.json (made on the GPU host by extract_curves.py).

    uv run python experiments/plot_curves.py

Writes returns_vs_gradient_steps.png, laptime_vs_gradient_steps.png and ppo_es_pipeline.png to
experiments/plots/.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).parent
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = {"PPO": "#2a78d6", "SAC": "#eb6834", "EPO": "#1baf7a", "ES": "#eda100"}
RUNS = {"PPO": "curve_ppo", "SAC": "curve_sac", "EPO": "curve_epo"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 11, "axes.titlesize": 12, "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False,
})
thousands = FuncFormatter(lambda x, _: f"{x / 1000:.0f}k" if x else "0")


def smooth(values, window=15):
    """Rolling median: the raw mean return spikes whenever the long standing-start episodes finish."""
    values = np.asarray(values)
    half = min(window, len(values) // 10) // 2
    return np.array([np.median(values[max(0, i - half):i + half + 1]) for i in range(len(values))])


def plot_returns(axes, x, y, color, label=None):
    """The raw mean return, faint, under its rolling median."""
    axes.plot(x, y, color=color, linewidth=0.8, alpha=0.35)
    axes.plot(x, smooth(y), color=color, label=label)


def returns_figure(curves):
    figure, axes = plt.subplots(figsize=(7, 4.5))
    for algorithm, run in RUNS.items():
        plot_returns(axes, curves[run]["return_gradient_steps"], curves[run]["return"], COLORS[algorithm], algorithm)
    axes.set(title="Training return", xlabel="Gradient steps", ylabel="Mean episode return", ylim=(0, None))
    axes.xaxis.set_major_formatter(thousands)
    axes.legend(loc="lower right")
    figure.tight_layout()
    return figure


def laptime_figure(curves):
    figure, axes = plt.subplots(figsize=(7, 4.5))
    for algorithm, run in RUNS.items():
        axes.plot(curves[run]["eval_gradient_steps"], curves[run]["eval_laps23"], color=COLORS[algorithm],
                  marker="o", markersize=4, label=algorithm)
    axes.set(title="Lap time", xlabel="Gradient steps", ylabel="Lap 2 + lap 3 time (s)", ylim=(None, 200))
    axes.xaxis.set_major_formatter(thousands)
    axes.legend(loc="upper right")
    figure.tight_layout()
    return figure


def pipeline_figure(curves):
    figure, panels = plt.subplots(1, 3, figsize=(13, 4))
    for axes, (run, title) in zip(panels[:2], (("curve_ppo", "PPO"), ("curve_ppo_ft", "PPO fine-tune"))):
        plot_returns(axes, curves[run]["return_gradient_steps"], curves[run]["return"], COLORS["PPO"])
        axes.set(title=title, xlabel="Gradient steps", ylabel="Mean episode return", ylim=(0, None))
        axes.xaxis.set_major_formatter(thousands)
    es, axes = curves["esS1_l2"], panels[2]
    axes.plot(es["real_generation"], es["real_laps23"], color=COLORS["ES"], marker="o", markersize=4)
    axes.set(title="ES", xlabel="ES updates", ylabel="Lap 2 + lap 3 time (s)")
    figure.tight_layout()
    return figure


def main():
    curves = json.loads((HERE / "curves" / "curves.json").read_text())
    output = HERE / "plots"
    output.mkdir(exist_ok=True)
    for name, make in (("returns_vs_gradient_steps", returns_figure), ("laptime_vs_gradient_steps", laptime_figure),
                       ("ppo_es_pipeline", pipeline_figure)):
        make(curves).savefig(output / f"{name}.png", dpi=200)
        print(output / f"{name}.png")


if __name__ == "__main__":
    main()
