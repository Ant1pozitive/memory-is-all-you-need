"""
Training script for copy and associative recall experiments.

Usage examples:
python -m src.train --task copy
"""
import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import cfg
from data.copy_dataset import CopyDataset
from model.memnet import MemNet
from utils.visualize import plot_slot_dynamics

def train_epoch(model, loader, opt, device, cfg):
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model.forward_sequence(inputs) # (B, T, V)
        B, T, V = logits.shape
        loss = ce(logits.view(B * T, V), targets.view(B * T))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        total_loss += loss.item() * inputs.size(0)
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    total_loss = 0.0
    total_correct = 0
    total_out = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model.forward_sequence(inputs)
        B, T, V = logits.shape
        preds = logits.argmax(-1)
        mask = targets != -100
        total_out += mask.sum().item()
        total_correct += (preds[mask] == targets[mask]).sum().item()
        total_loss += ce(logits.view(B * T, V), targets.view(B * T)).item()
    acc = total_correct / (total_out + 1e-12)
    return total_loss / len(loader.dataset), acc

def run_copy_experiment(save_dir: str):
    device = cfg.device
    os.makedirs(save_dir, exist_ok=True)

    train_ds = CopyDataset(cfg.vocab_size, cfg.seq_len, cfg.delay_len, size=5000)
    val_ds = CopyDataset(cfg.vocab_size, cfg.seq_len, cfg.delay_len, size=500)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    model = MemNet(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_acc = 0.0
    for ep in range(1, cfg.epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, device, cfg)
        val_loss, val_acc = evaluate(model, val_loader, device, cfg)
        print(f"Epoch {ep} | tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_memnet.pt'))
        print("Best val acc:", best_acc)

    # produce slot-dynamics visualization on a single example
    sample_in, sample_tgt = val_ds[0]
    sample_in = sample_in.unsqueeze(0).to(device)
    logits, read_w, write_w = model.forward_sequence(sample_in, return_attn=True)
    # read_w: (B, T, H, N)
    plot_slot_dynamics(read_w.cpu().numpy(), os.path.join(save_dir, 'read_dynamics.png'))
    print('Saved read_dynamics.png')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='copy', choices=['copy'])
    parser.add_argument('--out', type=str, default='runs/copy')
    args = parser.parse_args()

    if args.task == 'copy':
        run_copy_experiment(args.out)
