import pytest
import torch
from model.memory_bank import MultiHeadMemoryBank

@pytest.fixture
def memory_bank():
    return MultiHeadMemoryBank(num_slots=10, slot_dim=8, n_heads=2, topk=3)

def test_reset(memory_bank):
    mem = memory_bank.reset_memory(2, 'cpu')
    assert mem.shape == (2, 10, 8)

def test_read_write(memory_bank):
    B = 1
    mem = memory_bank.reset_memory(B, 'cpu')
    keys = torch.randn(B, 2, 8)
    vals = torch.randn(B, 2, 8)
    erase = torch.ones(B, 2) * 0.5
    add = torch.ones(B, 2)
    beta = torch.ones(2)
    new_mem, w = memory_bank.write(mem, keys, vals, erase, add, beta)
    read_vec, r_w = memory_bank.read(new_mem, keys, beta)
    assert read_vec.shape == (B, 8)

def test_lru(memory_bank_lru):
    memory_bank_lru = MultiHeadMemoryBank(10, 8, 2, 3, "lru")
    mem = memory_bank_lru.reset_memory(1, 'cpu')
    # Simulate writes/reads
    for _ in range(5):
        keys = torch.randn(1, 2, 8)
        vals = torch.randn(1, 2, 8)
        erase = torch.rand(1, 2)
        add = torch.rand(1, 2)
        beta = torch.ones(2)
        mem, _ = memory_bank_lru.write(mem, keys, vals, erase, add, beta)
    assert memory_bank_lru.access_times.max() > 0
