"""Exit this research process if another process starts using the selected GPU."""

import fcntl
import os
import subprocess
import threading
import time

import torch

_lock = None
_allowed = None
_device = None


def _pids():
    text = subprocess.check_output([
        "nvidia-smi", "-i", _device, "--query-compute-apps=pid", "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    return {line.strip() for line in text.splitlines() if line.strip()}


def check():
    foreign = _pids() - _allowed
    if foreign:
        raise RuntimeError(f"GPU {_device} now has other processes {sorted(foreign)}; stopping our work")


def start():
    global _lock, _allowed, _device
    if _allowed is not None:
        check()
        return
    _device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not _device or "," in _device:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to exactly one idle GPU")
    _lock = open(f"/tmp/mla_fp4_research_{_device}.lock", "a")
    fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Other containers' PIDs can be hidden while their memory is visible.
    memory = subprocess.check_output([
        "nvidia-smi", "-i", _device, "--query-gpu=memory.used", "--format=csv,noheader,nounits",
    ], text=True, timeout=10)
    if int(memory.strip()) > 64:
        raise RuntimeError(f"GPU {_device} has {memory.strip()} MiB already allocated; no GPU work launched")
    if _pids():
        raise RuntimeError(f"GPU {_device} is occupied; no GPU work launched")
    torch.cuda.init()
    torch.empty(1, device="cuda")
    torch.cuda.synchronize()
    _allowed = _pids()
    if len(_allowed) != 1:
        raise RuntimeError(f"Cannot establish sole ownership: {_allowed}")
    print(f"GPU guard: {_device}; our host PID={next(iter(_allowed))}", flush=True)

    def watch():
        while True:
            time.sleep(3)
            try:
                check()
            except Exception as error:
                print(str(error), flush=True)
                os._exit(75)  # Only this process; never other tasks or containers.

    threading.Thread(target=watch, daemon=True).start()
