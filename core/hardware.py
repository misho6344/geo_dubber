"""Hardware detection and device resolution (never assumes a GPU)."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


def _cuda_info() -> tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
        return False, ""
    except Exception:
        return False, ""


def _nvidia_smi_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return ""


def detect(path: str | Path = ".") -> dict:
    ram_total = ram_avail = None
    try:
        import psutil
        vm = psutil.virtual_memory()
        ram_total, ram_avail = vm.total, vm.available
    except Exception:
        pass
    cuda_ok, gpu_name = _cuda_info()
    if not gpu_name:
        gpu_name = _nvidia_smi_name()
    du = shutil.disk_usage(str(path))
    return {
        "cpu": platform.processor() or platform.machine(),
        "cpu_cores": os.cpu_count(),
        "ram_total_gb": round(ram_total / 2**30, 1) if ram_total else None,
        "ram_available_gb": round(ram_avail / 2**30, 1) if ram_avail else None,
        "gpu": gpu_name or "none detected",
        "cuda_available": cuda_ok,
        "disk_free_gb": round(du.free / 2**30, 1),
    }


def resolve_device(pref: str = "auto") -> tuple[str, str]:
    """Return (device, note). Falls back to CPU with an explanation instead of failing."""
    pref = (pref or "auto").lower()
    if pref == "cpu":
        return "cpu", ""
    cuda_ok, _ = _cuda_info()
    if cuda_ok:
        return "cuda", ""
    if pref == "cuda":
        return "cpu", "(CUDA requested but unavailable - using CPU)"
    return "cpu", "(no CUDA GPU - using CPU, processing will be slower)"


def check_disk_space(path: str | Path, needed_gb: float) -> None:
    from core.errors import InsufficientResourcesError
    free = shutil.disk_usage(str(path)).free / 2**30
    if free < needed_gb:
        raise InsufficientResourcesError(
            f"free {free:.1f} GB < needed {needed_gb:.1f} GB",
            f"Not enough disk space: {free:.1f} GB free, about {needed_gb:.1f} GB needed.")
