import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go
from torch.utils.tensorboard import SummaryWriter
from io import BytesIO
from PIL import Image
import os

def plot_attention_dynamics(weights: np.ndarray, out_path: str, title: str = "Attention Dynamics"):
    """
    Generate a static heatmap of attention dynamics.
    """
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
    """
    Generate a static PCA trajectory plot for memory slots.
    """
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

def interactive_attention_dynamics(weights: np.ndarray, out_path: str = "attention_dynamics.html", title: str = "Attention Dynamics"):
    """
    Generate an interactive heatmap of attention dynamics.
    """
    arr = weights
    if arr.ndim == 4:  # (B, T, H, N)
        arr = arr[0]
    T, H, N = arr.shape
    canvas = arr.transpose(2, 0, 1).reshape(N, T * H)
    
    fig = px.imshow(
        canvas,
        aspect='auto',
        color_continuous_scale='viridis',
        title=title
    )
    fig.update_layout(
        xaxis_title="Time × Head",
        yaxis_title="Memory Slot",
        height=600,
        width=1200
    )
    fig.write_html(out_path)

def interactive_slot_pca(memory_hist: np.ndarray, out_path: str = "slot_pca.html"):
    """
    Generate an interactive PCA trajectory plot for memory slots.
    """
    # memory_hist: (T, B, N, D) -> take first batch
    mem = memory_hist[:, 0]  # (T, N, D)
    pca = PCA(n_components=2)
    traj = pca.fit_transform(mem.reshape(-1, mem.shape[-1])).reshape(mem.shape[0], mem.shape[1], 2)
    
    fig = go.Figure()
    for slot in range(traj.shape[1]):
        fig.add_trace(go.Scatter(
            x=traj[:, slot, 0],
            y=traj[:, slot, 1],
            mode='lines+markers',
            name=f"Slot {slot}",
            opacity=0.7
        ))
    
    fig.update_layout(
        title="Memory Slot Trajectories (PCA)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.05),
        height=600,
        width=800,
        hovermode="closest"
    )
    fig.write_html(out_path)

def log_attention_dynamics_to_tensorboard(weights: np.ndarray, writer: SummaryWriter, global_step: int, title: str = "Attention Dynamics"):
    fig = plt.figure(figsize=(15, 8))
    arr = weights
    if arr.ndim == 4:
        arr = arr[0]
    T, H, N = arr.shape
    canvas = arr.transpose(2, 0, 1).reshape(N, T * H)
    plt.imshow(canvas, aspect='auto', cmap='viridis')
    plt.xlabel('Time × Head')
    plt.ylabel('Memory Slot')
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image).transpose(2, 0, 1)  # Convert to (C, H, W)
    writer.add_image(title, image, global_step)
    plt.close(fig)

def log_slot_pca_to_tensorboard(memory_hist: np.ndarray, writer: SummaryWriter, global_step: int):
    fig = plt.figure(figsize=(10, 8))
    mem = memory_hist[:, 0]  # (T, N, D)
    pca = PCA(n_components=2)
    traj = pca.fit_transform(mem.reshape(-1, mem.shape[-1])).reshape(mem.shape[0], mem.shape[1], 2)
    for slot in range(traj.shape[1]):
        plt.plot(traj[:, slot, 0], traj[:, slot, 1], label=f"Slot {slot}", alpha=0.7)
    plt.title("Memory Slot Trajectories (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = Image.open(buf)
    image = np.array(image).transpose(2, 0, 1)  # Convert to (C, H, W)
    writer.add_image("Memory Slot Trajectories (PCA)", image, global_step)
    plt.close(fig)

def log_visualizations_to_tensorboard(weights: np.ndarray,
                                      memory_hist: np.ndarray,
                                      log_dir: str = "logs",
                                      global_step: int = 0):
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    log_attention_dynamics_to_tensorboard(weights, writer, global_step)
    log_slot_pca_to_tensorboard(memory_hist, writer, global_step)
    writer.close()