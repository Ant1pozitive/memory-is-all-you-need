import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

import random
from config import cfg
from data.copy_dataset import CopyDataset
from model.memnet import MemNet
from utils.visualize import plot_attention_dynamics, plot_slot_dynamics

if cfg.train.use_wandb:
    import wandb
    wandb.init(project="memory-is-all-you-need", config=cfg.__dict__)

device = torch.device(cfg.device)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
        
set_seed(cfg.seed)

def get_dataset(task: str):
    if task == "copy":
        return CopyDataset(cfg.model.vocab_size, cfg.task.seq_len, cfg.task.delay_len, size=10000)
    # Add more tasks here later (continual, assoc, etc.)
    else:
        raise ValueError(f"Unknown task: {task}")

def sparsity_loss(weights: torch.Tensor) -> torch.Tensor:
    """Negative entropy to encourage sparsity"""
    eps = 1e-6
    return -torch.mean(weights * torch.log(weights + eps))

def diversity_loss(memory: torch.Tensor) -> torch.Tensor:
    """Encourage diverse slot contents (upper triangular cosine similarity)"""
    norm_mem = F.normalize(memory, dim=-1)
    sim = torch.bmm(norm_mem, norm_mem.transpose(1, 2)) # (B, N, N)
    return torch.mean(torch.triu(sim, diagonal=1))

def forgetting_loss(memory: torch.Tensor) -> torch.Tensor:
    """Memory pressure: penalize high norms on all slots -> encourage forgetting unused"""
    norms = memory.norm(dim=-1).mean(dim=1)
    return norms.mean()

def utilization_loss(weights: torch.Tensor) -> torch.Tensor:
    """
    Slot Utilization Loss: Penalizes low entropy in memory access.
    Encourages the model to explore and use more slots, preventing addressing collapse.
    """
    # weights shape: (B, H, N) or (B, T, H, N)
    avg_weights = weights.mean(dim=(0, 1)) # Average across batch and heads/time
    eps = 1e-8
    entropy = -torch.sum(avg_weights * torch.log(avg_weights + eps))
    # We want to maximize entropy, so we minimize negative entropy
    return -entropy

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, read_weights: torch.Tensor, memory: torch.Tensor):
    preds = logits.argmax(dim=-1)
    mask = targets != -100
    correct = (preds[mask] == targets[mask]).sum().item()
    total = mask.sum().item()
    acc = correct / total if total > 0 else 0.0
    output_positions = torch.cumsum(mask.float(), dim=1) - 1 # 0 to seq_len-1
    pos_correct = torch.zeros(cfg.task.seq_len, device=device)
    pos_count = torch.zeros(cfg.task.seq_len, device=device)
    for p in range(cfg.task.seq_len):
        p_mask = (output_positions == p) & mask
        pos_correct[p] = (preds[p_mask] == targets[p_mask]).sum()
        pos_count[p] = p_mask.sum()
    pos_acc = (pos_correct / (pos_count + 1e-8)).cpu().numpy()
    slot_norms = memory.norm(dim=-1).mean(dim=0)
    utilization = (slot_norms > 0.1).float().mean().item()
    read_sparsity = (read_weights > 0.01).float().mean().item()
    return acc, pos_acc, utilization, read_sparsity

def train_epoch(model: MemNet, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scaler: GradScaler, epoch: int):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_ce = 0.0
    total_aux = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        with autocast(enabled=cfg.train.mixed_precision):
                    logits, read_w, write_w = model(inputs, return_attn=True)
                   
                    # Calculate standard losses
                    ce_loss = ce_loss_fn(logits.view(-1, cfg.model.vocab_size), targets.view(-1))
                   
                    # Slot Utilization Loss
                    util_loss = utilization_loss(read_w) + utilization_loss(write_w)
                   
                    # Aux losses
                    spar_loss = sparsity_loss(read_w) + sparsity_loss(write_w)
                   
                    # Total loss
                    loss = ce_loss + (cfg.train.lambda_sparsity * spar_loss) + \
                        (cfg.train.lambda_utilization * util_loss)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_ce += ce_loss.item() * batch_size
        total_aux += spar_loss.item() * batch_size
        pbar.set_postfix({"loss": loss.item(), "ce": ce_loss.item()})
    n = len(loader.dataset)
    metrics = {
        "train/loss": total_loss / n,
        "train/ce_loss": total_ce / n,
        "train/aux_loss": total_aux / n,
    }
    if cfg.train.use_wandb:
        wandb.log(metrics, step=epoch)
    return total_loss / n

@torch.no_grad()
def evaluate(model: MemNet, loader: DataLoader, epoch: int):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    total_loss = 0.0
    total_correct = 0
    total_tokens = 0
    all_pos_acc = np.zeros(cfg.task.seq_len)
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)
        with autocast(enabled=cfg.train.mixed_precision):
            logits, read_w, write_w = model(inputs, return_attn=True)
        loss = ce_loss_fn(logits.view(-1, cfg.model.vocab_size), targets.view(-1))
        total_loss += loss.item()
        acc, pos_acc, util, spar = compute_metrics(logits, targets, read_w, model.memory.init_memory) # memory approx
        total_correct += acc * targets.size(0)
        total_tokens += targets.size(0)
        all_pos_acc += pos_acc
        pbar.set_postfix({"val_loss": loss.item(), "acc": acc})
    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_tokens
    avg_pos_acc = all_pos_acc / len(loader)
    metrics = {
        "val/loss": avg_loss,
        "val/accuracy": avg_acc,
        "val/memory_utilization": util,
        "val/read_sparsity": spar,
    }
    if cfg.train.use_wandb:
        wandb.log(metrics, step=epoch)
        wandb.log({f"val/pos_acc_{i}": avg_pos_acc[i] for i in range(len(avg_pos_acc))}, step=epoch)
    return avg_loss, avg_acc

def main(args):
    os.makedirs(args.out, exist_ok=True)
    train_dataset = get_dataset(args.task)
    val_dataset = get_dataset(args.task)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    model = MemNet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scaler = GradScaler(enabled=cfg.train.mixed_precision)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    best_acc = 0.0
    patience_counter = 0
    for epoch in range(1, cfg.train.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scaler, epoch)
        val_loss, val_acc = evaluate(model, val_loader, epoch)
        scheduler.step(val_loss)
        print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
            patience_counter = 0
            print(" -> New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print("Early stopping triggered.")
                break
    print(f"Training finished. Best validation accuracy: {best_acc:.4f}")
   
    model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pt")))
    model.eval()
    sample_inputs, sample_targets = next(iter(val_loader))
    sample_inputs = sample_inputs[:1].to(device)
    with torch.no_grad():
        with autocast(enabled=cfg.train.mixed_precision):
            logits, read_w, write_w = model(sample_inputs, return_attn=True)
    read_w = read_w.cpu().numpy()
    write_w = write_w.cpu().numpy()
    plot_attention_dynamics(read_w, os.path.join(args.out, "read_dynamics.png"), "Read Attention")
    plot_attention_dynamics(write_w, os.path.join(args.out, "write_dynamics.png"), "Write Attention")
    print(f"Visualizations saved to {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Memory-Augmented Network")
    parser.add_argument("--task", type=str, default="copy", choices=["copy"])
    parser.add_argument("--out", type=str, default="runs/experiment_001", help="Output directory")
    args = parser.parse_args()
    main(args)