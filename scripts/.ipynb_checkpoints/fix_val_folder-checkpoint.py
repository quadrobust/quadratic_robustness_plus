#!/usr/bin/env python
import pathlib, shutil, sys

VAL_DIR = pathlib.Path("data/imagenet")
GT_FILE = pathlib.Path("data/imagenet/ILSVRC2012_devkit_t12/data"
                       "/ILSVRC2012_validation_ground_truth.txt")

# 1) wczytaj listę 50k wnid‑ów (dodaj prefiks "n")
wnids = ["n" + line.strip() for line in GT_FILE.open()]
assert len(wnids) == 50000, "ground‑truth nie ma 50k wierszy"

# 2) posortuj pliki alfabetycznie
files = sorted(VAL_DIR.glob("*.JPEG"))
assert len(files) == 50000, f"znalazłem {len(files)} plików, powinno być 50k"

print("→ przenoszę pliki… (to potrwa < 20 s na NVMe)")

for src, wnid in zip(files, wnids):
    dst_dir = VAL_DIR / wnid
    dst_dir.mkdir(exist_ok=True)
    shutil.move(str(src), dst_dir / src.name)

print("✔️  Gotowe: pliki w 1000 podfolderach.")
