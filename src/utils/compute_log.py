# src/utils/compute_log.py
"""
ComputeLogger: context manager to log runtime and GPU memory usage per code block.

Usage:
    with ComputeLogger(tag="mytag", extra={"eps": 0.3}):
        # run evaluation or training

Writes one JSONL entry per exit to metrics/compute_log.jsonl:
    {
      "tag": "mytag",
      "runtime_s": 12.34,
      "gpu_name": "NVIDIA A100",
      "gpu_count": 1,
      "peak_mem_MB": 1234.5,
      "eps": 0.3
    }
"""

import time, json, os, torch
import sys

_LOGFILE = "metrics/compute_log.jsonl"

class ComputeLogger:
    def __init__(self, tag, extra=None):
        self.tag   = tag
        self.extra = extra or {}
    def __enter__(self):
        torch.cuda.reset_peak_memory_stats()
        self.start = time.time()
        return self
    def __exit__(self, exc_type, exc, tb):
        end = time.time()
        entry = {
            "tag":        self.tag,
            "runtime_s":  round(end - self.start, 2),
            "gpu_name":   torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "gpu_count":  torch.cuda.device_count(),
            "peak_mem_MB": round(torch.cuda.max_memory_allocated() / 2**20, 1) \
                           if torch.cuda.is_available() else 0,
            **self.extra
        }
        os.makedirs(os.path.dirname(_LOGFILE), exist_ok=True)
        with open(_LOGFILE, "a") as f:
            f.write(json.dumps(entry) + "\n")
        #print(f"[ComputeLog] {entry}")
