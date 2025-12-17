import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_attention_dynamics(weights, out_path: str, title: str, head: int = 0):
    arr = np.array(weights)
    if arr.ndim == 4:
        arr = arr[0]
    T, H, N = arr.shape
    canvas = arr.transpose(2, 0, 1).reshape(N, T * H)
    plt.figure(figsize=(12, 6))
    plt.imshow(canvas, aspect='auto', cmap='viridis')
    plt.xlabel('time x head')
    plt.ylabel('memory slot')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def plot_slot_dynamics(read_weights, write_weights, memory_hist, out_path: str):
    plot_attention_dynamics(read_weights, out_path.replace('.png', '_read.png'), 'Read attention dynamics')
    plot_attention_dynamics(write_weights, out_path.replace('.png', '_write.png'), 'Write attention dynamics')
    # PCA on slots over time
    memory_np = np.array(memory_hist)  # (T, B, N, D) -> take [0]
    if memory_np.ndim == 4:
        memory_np = memory_np[:, 0]  # (T, N, D)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(memory_np.reshape(-1, memory_np.shape[-1])).reshape(memory_np.shape[0], memory_np.shape[1], 2)
    plt.figure(figsize=(12, 6))
    for slot in range(proj.shape[1]):
        plt.plot(proj[:, slot, 0], proj[:, slot, 1], label=f'Slot {slot}')
    plt.title('Slot contents PCA trajectory')
    plt.legend()
    plt.savefig(out_path.replace('.png', '_slots_pca.png'))
    plt.close()
