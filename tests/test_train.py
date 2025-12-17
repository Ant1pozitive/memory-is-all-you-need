import pytest
import torch
from model.memnet import MemNet
from config import cfg
from train import train_epoch, evaluate
from torch.utils.data import DataLoader
from data.copy_dataset import CopyDataset

@pytest.fixture
def model():
    cfg.train.batch_size = 2
    cfg.train.epochs = 1
    cfg.task.seq_len = 2
    cfg.task.delay_len = 2
    return MemNet(cfg).to('cpu')

def test_train_eval(model):
    ds = CopyDataset(cfg.model.vocab_size, cfg.task.seq_len, cfg.task.delay_len)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    tr_loss = train_epoch(model, loader, opt, scaler, 'cpu', cfg, 1)
    assert tr_loss > 0
    val_loss, acc, pos_acc = evaluate(model, loader, 'cpu', cfg)
    assert val_loss > 0
    assert len(pos_acc) == cfg.task.seq_len
