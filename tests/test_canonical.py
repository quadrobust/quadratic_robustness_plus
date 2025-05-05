import torch

import sys, os

import numpy as np
# 1) dodajemy katalog główny projektu na początek ścieżki importów
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import torch, random, inspect, textwrap
from src.geometry.canonical_forms import FORMS, get_form, random_form

x = torch.zeros(1,1);  y = torch.zeros(1,1)

def test_lengths():
    """Każda zarejestrowana orbita ma poprawną długość."""
    for (n, id_) in FORMS:
        out = get_form(n, id_)(x, y)
        assert len(out) == n, f"Orbit {(n,id_)}: len={len(out)} ≠ {n}"

def test_pad_high_dim():
    """Pad‑zeros działa dla n>5."""
    out = get_form(8, 0)(x, y)     # generic z zerami
    assert len(out) == 8

def _src_of(fn):
    """Zwraca jednolinijkowy kod λ‑funkcji."""
    try:
        src = inspect.getsource(fn)
        # usuń 'lambda x,y:' i przecinki koncowe
        body = src.split("lambda",1)[1].split(":",1)[1].strip()
        return body.replace('\n','').replace(' ','')
    except (OSError, IOError):
        return "<source‑unavailable>"

def test_random_form_sampling():
    rng = random.Random(123)
    for n in range(1,6):
        pool = [(id_,f) for (nn,id_),f in FORMS.items() if nn==n]
        assert pool, f"No forms for n={n}"
        print(f"\n— n={n} —")
        for k in range(3):
            fn = random_form(n, rng)
            # identyfikacja id
            id_match = next(id_ for (id_,f) in pool if f is fn)
            print(f"  • id={id_match:<3}  {_src_of(fn)}")
            assert len(fn(x,y)) == n
    print("\n")  # pusty wiersz dla czytelności
    #capfd.readouterr()      # wymuś wypis na stdout
