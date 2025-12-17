import argparse
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np

from config import cfg
from data.copy_dataset import CopyDataset
from data.associative_recall import AssociativeRecallDataset
from model.memnet import MemNet
from utils.visualize import plot_slot_dynamics

def get_dataset(task):
    if task == "copy":
        return CopyDataset(cfg.model.vocab_size, cfg.task.seq_len, cfg.task.delay_len)
    elif task == "assoc":
        return AssociativeRecallDataset(cfg.model.vocab_size, cfg.task.assoc_pairs, cfg.task.delay_len)
    raise ValueError("Unknown task")

def sparsity_loss(weights):
    return -torch.mean(weights * torch.log(weights + 1e-6))  # entropy

def diversity_loss(memory):
    sim = torch.einsum('bnd,bmd->bnm', memory, memory)
    return torch.mean(sim.triu(1))  # upper tri mean sim

def usage_loss(weights):
    return -torch.mean(torch.sum(weights, dim=-1))  # encourage usage

def train_epoch(model, loader, opt, device, cfg, epoch):
    model.train()
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    total_aux = 0.0
    for inputs, targets in tqdm(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits, read_w, write_w = model.forward_sequence(inputs, return_attn=True)
        B, T, V = logits.shape
        ce_loss = ce(logits.view(B * T, V), targets.view(B * T))
        spar_loss = sparsity_loss(read_w.mean(1)) + sparsity_loss(write_w.mean(1))
        div_loss = diversity_loss(model.memory.init_memory)  # approx
        use_loss = usage_loss(read_w.mean(1)) + usage_loss(write_w.mean(1))
        aux_loss = cfg.train.lambda_sparsity * spar_loss + cfg.train.lambda_diversity * div_loss + cfg.train.lambda_usage * use_loss
        loss = ce_loss + aux_loss
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        total_loss += ce_loss.item() * inputs.size(0)
        total_aux += aux_loss.item() * inputs.size(0)
    if cfg.train.use_wandb:
        wandb.log({"train_loss": total_loss / len(loader.dataset), "aux_loss": total_aux / len(loader.dataset), "epoch": epoch})
    return total_loss / len(loader.dataset)

@torch.no_grad()
def evaluate(model, loader, device, cfg):
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=-100, reduction='sum')
    total_loss = 0.0
    total_correct = 0
    total_out = 0
    pos_correct = np.zeros(cfg.task.seq_len)  # positional acc
    pos_count = np.zeros(cfg.task.seq_len)
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model.forward_sequence(inputs)
        B, T, V = logits.shape
        preds = logits.argmax(-1)
        mask = targets != -100
        out_pos = torch.cumsum(mask.float(), dim=1) - 1  # positions in output
        for p in range(cfg.task.seq_len):
            p_mask = (out_pos == p) & mask
            pos_correct[p] += (preds[p_mask] == targets[p_mask]).sum().item()
            pos_count[p] += p_mask.sum().item()
        total_out += mask.sum().item()
        total_correct += (preds[mask] == targets[mask]).sum().item()
        total_loss += ce(logits.view(B * T, V), targets.view(B * T)).item()
    acc = total_correct / (total_out + 1e-12)
    pos_acc = pos_correct / (pos_count + 1e-12)
    if cfg.train.use_wandb:
        wandb.log({"val_acc": acc, "val_loss": total_loss / len(loader.dataset)})
    return total_loss / len(loader.dataset), acc, pos_acc

def run_experiment(task: str, save_dir: str):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = cfg.device
    os.makedirs(save_dir, exist_ok=True)
    if cfg.train.use_wandb:
        wandb.init(project="memory-net", config=cfg.__dict__)

    train_ds = get_dataset(task)
    val_ds = get_dataset(task)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    model = MemNet(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)
    best_acc = 0.0
    patience_cnt = 0

    for ep in range(1, cfg.train.epochs + 1):
        tr_loss = train_epoch(model, train_loader, opt, device, cfg, ep)
        val_loss, val_acc, pos_acc = evaluate(model, val_loader, device, cfg)
        print(f"Epoch {ep} | tr_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | pos_acc={pos_acc}")
        scheduler.step(val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_memnet.pt'))
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.train.patience:
                print("Early stopping")
                break

    print("Best val acc:", best_acc)

    sample_in, sample_tgt = next(iter(val_loader))
    sample_in = sample_in[0:1].to(device)
    logits, read_w, write_w = model.forward_sequence(sample_in, return_attn=True)
    memory_hist = []
    plot_slot_dynamics(read_w.cpu().numpy(), write_w.cpu().numpy(), memory_hist, os.path.join(save_dir, 'dynamics.png'))
    print('Saved dynamics plots')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='copy', choices=['copy', 'assoc'])
    parser.add_argument('--out', type=str, default='runs/copy')
    args = parser.parse_args()
    run_experiment(args.task, args.out)
