import numpy as np
import plotext as plt
from numpy.typing import NDArray


def plot_solution(x_known: NDArray, best_idx: NDArray, top_size: int) -> None:
    """Plot (sorted) known solution along with best indices provided."""
    # Squeeze array
    x_known = np.squeeze(x_known)

    # Find mapping from index in sorted array to original index
    sorted_to_original_idx = np.flip(np.argsort(x_known))

    # Sort in descending order
    sorted_x_known = x_known[sorted_to_original_idx]

    # Find mapping from original index to index in sorted array
    original_to_sorted_idx = np.zeros(x_known.size, dtype=int)
    original_to_sorted_idx[sorted_to_original_idx] = range(x_known.size)

    # Plot
    subset = 3 * top_size
    plt.bar(range(sorted_x_known.size), sorted_x_known[:subset], color="skyblue")
    plt.bar(
        original_to_sorted_idx[best_idx][:subset],
        sorted_x_known[original_to_sorted_idx[best_idx]][:subset],
        color="red",
    )
    plt.show()
