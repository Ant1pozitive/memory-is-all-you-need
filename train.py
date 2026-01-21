import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import cfg
from data.copy_dataset import CopyDataset
from model.memnet import MemNet

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

def sparsity_loss(weights):
    return - (weights * torch.log(weights + 1e-8)).sum(-1).mean()

def utilization_loss(weights):
    return - (- (weights.mean((0,1)) * torch.log(weights.mean((0,1)) + 1e-8)).sum())

def graph_sparsity_loss(adjacency):
    """
    Penalizes dense graphs. Encourages efficient, sparse topology.
    L1 norm of the adjacency matrix.
    """
    return adjacency.mean()

def train_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss_fn = nn.MSELoss()
    
    for inputs, targets in tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False):
        inputs, targets = inputs.to(cfg.device), targets.to(cfg.device)
        optimizer.zero_grad()
        
        with torch.amp.autocast(device_type=device_type, enabled=cfg.train.mixed_precision):
            # Now forward returns adjacency tuple as well
            logits, recon, (read_w_stm, read_w_ltm), write_w, (adj_stm, adj_ltm) = model(inputs, return_attn=True)
            
            # 1. Task Loss
            task_loss = ce_loss_fn(logits.view(-1, cfg.model.vocab_size), targets.view(-1))
            
            # 2. Hallucination Loss
            with torch.no_grad():
                target_emb = model.controller.embed(inputs)
            hall_loss = mse_loss_fn(recon, target_emb)
            
            # 3. Structural Losses
            spar_loss = sparsity_loss(read_w_stm) + sparsity_loss(read_w_ltm) + sparsity_loss(write_w)
            util_loss = utilization_loss(read_w_stm) + utilization_loss(read_w_ltm) + utilization_loss(write_w)
            
            # 4. Topological Graph Regularization
            graph_loss = graph_sparsity_loss(adj_stm) + graph_sparsity_loss(adj_ltm)
            
            total_loss = task_loss + \
                         cfg.train.lambda_hallucination * hall_loss + \
                         cfg.train.lambda_sparsity * spar_loss + \
                         cfg.train.lambda_utilization * util_loss + \
                         cfg.train.lambda_graph_sparsity * graph_loss
        
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
        
        with torch.amp.autocast(device_type=device_type, enabled=cfg.train.mixed_precision):
            # Ignore extra returns for eval
            logits, _, _, _, _ = model(inputs, return_attn=True)
        
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
    scaler = torch.amp.GradScaler(device_type, enabled=cfg.train.mixed_precision)
    
    for epoch in range(1, cfg.train.epochs + 1):
        train_epoch(model, train_loader, optimizer, scaler, epoch)
        val_loss, val_acc = evaluate(model, val_loader, epoch)
        print(f"Epoch {epoch}: Val Loss {val_loss:.4f}, Val Acc {val_acc:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="copy")
    args = parser.parse_args()
    main(args)