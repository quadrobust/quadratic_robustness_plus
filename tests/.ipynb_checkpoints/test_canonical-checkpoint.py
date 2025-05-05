import torch
from src.canonical_forms import FORMS, get_form

def test_lengths():
    x = torch.zeros(1,1)
    y = torch.zeros(1,1)
    for (n,id_) in FORMS:
        f = get_form(n,id_)
        out = f(x,y)
        assert len(out) == n, f"Orbit {(n,id_)} zwraca len={len(out)}"

def test_pad_high_dim():
    x = torch.zeros(1,1)
    y = torch.zeros(1,1)
    f = get_form(8,0)     # generic + zeros
    assert len(f(x,y)) == 8

