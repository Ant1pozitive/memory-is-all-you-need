import matplotlib.pyplot as plt
import numpy as np

def plot_slot_dynamics(read_weights, out_path: str, head: int = 0):
    """
    read_weights: (T, H, N) or (B, T, H, N) - we accept either and plot first batch if present
    Saves heatmap to out_path
    """
    arr = np.array(read_weights)
    if arr.ndim == 4:
        arr = arr[0] # take first batch -> (T, H, N)
    # transpose to (N, T*H) or plot per head stacked horizontally
    T, H, N = arr.shape
    # we'll plot concatenated heads: horizontally time*H, vertically N
    canvas = arr.transpose(2, 0, 1).reshape(N, T * H)
    plt.figure(figsize=(12, 6))
    plt.imshow(canvas, aspect='auto', cmap='viridis')
    plt.xlabel('time x head')
    plt.ylabel('memory slot')
    plt.title('Read attention dynamics (slots x time*heads)')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
