import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_attention_dynamics(weights: np.ndarray, out_path: str, title: str = "Attention Dynamics"):
    arr = weights
    if arr.ndim == 4:  # (B, T, H, N)
        arr = arr[0]
    T, H, N = arr.shape
    canvas = arr.transpose(2, 0, 1).reshape(N, T * H)
    plt.figure(figsize=(15, 8))
    plt.imshow(canvas, aspect='auto', cmap='viridis')
    plt.xlabel('Time × Head')
    plt.ylabel('Memory Slot')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_slot_pca(memory_hist: np.ndarray, out_path: str):
    # memory_hist: (T, B, N, D) -> take first batch
    mem = memory_hist[:, 0]  # (T, N, D)
    pca = PCA(n_components=2)
    traj = pca.fit_transform(mem.reshape(-1, mem.shape[-1])).reshape(mem.shape[0], mem.shape[1], 2)

    plt.figure(figsize=(10, 8))
    for slot in range(traj.shape[1]):
        plt.plot(traj[:, slot, 0], traj[:, slot, 1], label=f"Slot {slot}", alpha=0.7)
    plt.title("Memory Slot Trajectories (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
