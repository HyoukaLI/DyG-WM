from __future__ import annotations

import gc
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch (all devices)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(gpu: int | str | None) -> torch.device:
    """``--gpu 0`` -> cuda:0; ``--gpu -1`` -> CPU; ``auto`` -> CUDA, MPS, CPU."""
    if gpu is None or str(gpu) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if int(gpu) < 0:
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(f"--gpu {gpu} was requested but CUDA is not available")
    return torch.device(f"cuda:{int(gpu)}")


def device_description(device: torch.device) -> str:
    if device.type == "cuda":
        index = torch.cuda.current_device() if device.index is None else device.index
        return f"cuda:{index} ({torch.cuda.get_device_name(index)})"
    return str(device)


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def release_device_memory(device: torch.device) -> None:
    """Best-effort release of cached accelerator memory between runs."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def create_logger(name: str, log_file: Path) -> logging.Logger:
    """Logger writing to both the console and ``log_file``."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
