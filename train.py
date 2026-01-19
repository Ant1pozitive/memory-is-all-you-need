import argparse
import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config import cfg
from data.copy_dataset import CopyDataset
from model.memnet import MemNet
from utils.visualize import plot_attention_dynamics

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(cfg.seed)
np.random.seed(cfg.seed)

def sparsity_loss(weights: torch.Tensor) -> torch.Tensor:
    return - (weights * torch.log(weights + 1e-8)).sum(-1).mean()

def utilization_loss(weights: torch.Tensor) -> torch.Tensor:
    return - (- (weights.mean((0,1)) * torch.log(weights.mean((0,1)) + 1e-8)).sum())

def train_epoch(model, loader, optimizer, scaler, epoch):
    """
    Performs one training epoch with explicit loss logging and comments.
    """
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    
    for inputs, targets in tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False):
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        optimizer.zero_grad()
        
        # Using Mixed Precision for faster training
        with torch.amp.autocast(device_type=device_type, enabled=cfg.train.mixed_precision):
            # Forward pass returns:
            # logits: model predictions [B, T, Vocab]
            # recon: reconstructed embeddings [B, T, Embed]
            # read_w_tuple: (STM_weights, LTM_weights)
            # write_w: weights used for writing to STM
            logits, recon, (read_w_stm, read_w_ltm), write_w = model(inputs, return_attn=True)
            
            # --- 1. Task Loss (Categorical Cross-Entropy) ---
            # Standard loss for token prediction
            task_loss = ce_loss_fn(logits.view(-1, cfg.model.vocab_size), targets.view(-1))
            
            # --- 2. Hallucination Loss ---
            # We compare the 'recon' (what model thinks it sees based on memory)
            # with the actual 'target_emb' (original embeddings of input tokens).
            with torch.no_grad():
                # Get actual embeddings of the input tokens to use as ground truth
                # Shape: [Batch, Time, Embed_dim]
                target_emb = model.controller.embed(inputs)
            
            # We compare Step-by-Step [B, T, E] with [B, T, E]
            # No more mean(dim=1), so dimensions match (121 == 121)
            hall_loss = mse_loss_fn(recon, target_emb)
            
            # --- 3. Structural Losses (Sparsity & Utilization) ---
            # These help memory organize itself efficiently
            spar_loss = sparsity_loss(read_w_stm) + sparsity_loss(read_w_ltm) + sparsity_loss(write_w)
            util_loss = utilization_loss(read_w_stm) + utilization_loss(read_w_ltm) + utilization_loss(write_w)
            
            # Total weighted loss
            total_loss = task_loss + \
                         cfg.train.lambda_hallucination * hall_loss + \
                         cfg.train.lambda_sparsity * spar_loss + \
                         cfg.train.lambda_utilization * util_loss
        
        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        scaler.step(optimizer)
        scaler.update()

def evaluate(model, loader, epoch):
    model.eval()
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    total_loss = 0.0
    correct = total = 0
    
    for inputs, targets in tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False):
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        
        with torch.amp.autocast(device_type=cfg.device.split(':')[0], enabled=cfg.train.mixed_precision):
            logits, _, _, _ = model(inputs, return_attn=True)
        
        loss = ce_loss_fn(logits.view(-1, cfg.model.vocab_size), targets.view(-1))
        total_loss += loss.item()
        
        pred = logits.argmax(-1)
        mask = targets != -100
        correct += (pred[mask] == targets[mask]).sum().item()
        total += mask.sum().item()
    
    return total_loss / len(loader), correct / total if total > 0 else 0

def main(args):
    train_dataset = CopyDataset(split='train', cfg=cfg)
    val_dataset = CopyDataset(split='val', cfg=cfg)
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False)
    
    model = MemNet(cfg).to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr)
    scaler = torch.amp.GradScaler(cfg.device.split(':')[0], enabled=cfg.train.mixed_precision)
    
    for epoch in range(1, cfg.train.epochs + 1):
        train_epoch(model, train_loader, optimizer, scaler, epoch)
        val_loss, val_acc = evaluate(model, val_loader, epoch)
        print(f"Epoch {epoch}: Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")
        
        if epoch % 5 == 0:
            sample_input, _ = next(iter(val_loader))
            _, _, (r_w_stm, r_w_ltm), write_w = model(sample_input.to(cfg.device), return_attn=True)
            # Plot only STM for now to maintain visualization compatibility
            plot_attention_dynamics(r_w_stm[0], write_w[0], epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="copy")
    args = parser.parse_args()
    main(args)