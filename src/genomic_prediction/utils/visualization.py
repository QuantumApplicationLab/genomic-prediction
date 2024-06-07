import numpy as np
import plotext as plt
from numpy.typing import NDArray


def plot_solution(x_known: NDArray, best_idx: NDArray, top_size: int) -> None:
    """Plot (sorted) known solution along with best indices provided."""
    # Squeeze array
    x_known = np.squeeze(x_known)

    # Define colors for plotting
    colors = np.array(x_known.size * ["blue"])
    colors[best_idx] = "red"

    # Find mapping from index in sorted array to original index
    sort_idx = np.flip(np.argsort(x_known))

    # Sort in descending order
    sorted_x_known = x_known[sort_idx]
    sorted_colors = colors[sort_idx]

    # Plot
    ignore = 4
    subset = 4 * top_size
    plt.plot(sorted_x_known[ignore:subset])
    for idx, value in enumerate(sorted_colors[ignore:subset]):
        if value == "red":
            plt.vline(idx, value)
    plt.show()
