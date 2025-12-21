import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from config import cfg
from data.copy_dataset import CopyDataset
from model.memnet import MemNet
from utils.visualize import plot_attention_dynamics

if cfg.train.use_wandb:
    import wandb
    wandb.init(project="memory-is-all-you-need", config=cfg.__dict__)

device = torch.device(cfg.device)
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

def get_dataset(task: str):
    if task == "copy":
        return CopyDataset(cfg.model.vocab_size, cfg.task.seq_len, cfg.task.delay_len, size=10000)
    else:
        raise ValueError(f"Unknown task: {task}")

def sparsity_loss(weights: torch.Tensor) -> torch.Tensor:
    """Encourages sparse memory access."""
    eps = 1e-6
    return -torch.mean(weights * torch.log(weights + eps))

def utilization_loss(weights: torch.Tensor) -> torch.Tensor:
    """Slot Utilization Loss: Penalizes low entropy (collapse) in memory access."""
    avg_weights = weights.mean(dim=(0, 1))
    eps = 1e-8
    entropy = -torch.sum(avg_weights * torch.log(avg_weights + eps))
    return -entropy

def train_epoch(model: MemNet, loader: DataLoader, optimizer: torch.optim.Optimizer,
                scaler: GradScaler, epoch: int):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    # We will use the model's embedding layer to get targets for reconstruction
    # A bit of a hack, but efficient.
    embed_layer = model.controller.embed 
    
    total_loss = 0.0
    total_ce = 0.0
    total_hal = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast(enabled=cfg.train.mixed_precision):
            # Forward pass returns logits and reconstruction vectors
            logits, recon_vecs, read_w, write_w = model(inputs, return_attn=True)
            
            # 1. Main Task Loss
            ce_loss = ce_loss_fn(logits.view(-1, cfg.model.vocab_size), targets.view(-1))
            
            # 2. Hallucination (Reconstruction) Loss
            with torch.no_grad():
                target_embeds = embed_layer(inputs) # [B, T, E]
            hal_loss = F.mse_loss(recon_vecs, target_embeds)
            
            # 3. Aux Losses
            spar_loss = sparsity_loss(read_w) + sparsity_loss(write_w)
            util_loss = utilization_loss(read_w) + utilization_loss(write_w)

            loss = ce_loss + \
                   (cfg.train.lambda_hallucination * hal_loss) + \
                   (cfg.train.lambda_sparsity * spar_loss) + \
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
        total_hal += hal_loss.item() * batch_size

        pbar.set_postfix({"loss": loss.item(), "ce": ce_loss.item(), "hal": hal_loss.item()})

    n = len(loader.dataset)
    metrics = {
        "train/loss": total_loss / n,
        "train/ce_loss": total_ce / n,
        "train/hal_loss": total_hal / n,
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

    pbar = tqdm(loader, desc=f"Epoch {epoch} [Val]")
    for inputs, targets in pbar:
        inputs = inputs.to(device)
        targets = targets.to(device)

        with autocast(enabled=cfg.train.mixed_precision):
            logits, _, _, _ = model(inputs, return_attn=True)

        loss = ce_loss_fn(logits.view(-1, cfg.model.vocab_size), targets.view(-1))
        total_loss += loss.item()

        preds = logits.argmax(dim=-1)
        mask = targets != -100
        total_correct += (preds[mask] == targets[mask]).sum().item()
        total_tokens += mask.sum().item()

        pbar.set_postfix({"val_loss": loss.item()})

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_tokens if total_tokens > 0 else 0.0

    if cfg.train.use_wandb:
        wandb.log({"val/loss": avg_loss, "val/accuracy": avg_acc}, step=epoch)

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
        
        print(f"Ep {epoch:03d} | Tr: {train_loss:.4f} | Val: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.out, "best_model.pt"))
            patience_counter = 0
            print("--> New best model saved!")
        else:
            patience_counter += 1
            if patience_counter >= cfg.train.patience:
                print("Early stopping.")
                break
    
    # Save visualizations for the best model
    model.load_state_dict(torch.load(os.path.join(args.out, "best_model.pt")))
    model.eval()
    sample_inputs, _ = next(iter(val_loader))
    sample_inputs = sample_inputs[:1].to(device)
    
    with torch.no_grad():
        _, _, read_w, write_w = model(sample_inputs, return_attn=True)
        
    plot_attention_dynamics(read_w.cpu().numpy(), os.path.join(args.out, "read_pattern.png"), "Read Attention")
    plot_attention_dynamics(write_w.cpu().numpy(), os.path.join(args.out, "write_pattern.png"), "Write Attention")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="copy")
    parser.add_argument("--out", type=str, default="runs/experiment_hebbian")
    args = parser.parse_args()
    main(args)