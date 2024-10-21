"""Helpers for creating and saving plots."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.size"] = "4"


def _get_plot_dir():
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    return plots_dir


def save_fig(name: str):
    """Save figure."""
    plots_dir = _get_plot_dir()
    plt.savefig(Path(plots_dir, name), dpi=600, bbox_inches="tight")
    plt.close("all")
