import pytest
from src.data.copy_dataset import CopyDataset

@pytest.fixture
def ds():
    return CopyDataset(vocab_size=10, seq_len=3, delay_len=2)

def test_getitem(ds):
    inp, tgt = ds[0]
    assert len(inp) == 3 + 2 + 1 + 3
    assert (tgt[-3:] == inp[:3]).all()
