from __future__ import annotations

import copy
from collections import OrderedDict

import torch
from torch import nn


def quantize_tensor(
    tensor: torch.Tensor,
    nbits: int = 4,
    block_size: int = 64,
) -> dict[str, torch.Tensor | tuple[int, ...] | int | torch.dtype | bool]:
    """Quantize a PyTorch tensor into scaled integer blocks with sub-byte packing.

    Supports INT4 (sub-byte packed into torch.uint8 nibbles) and INT8 (torch.int8)
    with per-block dynamic scaling. Zero blocks and non-finite values are handled safely.

    Args:
        tensor: Floating-point PyTorch tensor of arbitrary shape.
        nbits: Quantization bit-width (4 or 8).
        block_size: Number of elements per quantization block (even positive integer).

    Returns:
        Dictionary containing packed data, per-block scale factors, original shape,
        padding length, and metadata necessary for exact reconstruction.
    """
    if nbits not in (4, 8):
        raise ValueError(f"nbits must be 4 or 8, got {nbits}")
    if block_size < 2 or block_size % 2 != 0:
        raise ValueError(
            f"block_size must be a positive even integer, got {block_size}"
        )

    shape = tuple(tensor.shape)
    orig_dtype = tensor.dtype
    device = tensor.device
    flat = tensor.detach().flatten().float()
    numel = flat.numel()

    if numel == 0:
        empty_data = torch.empty(
            0, dtype=torch.uint8 if nbits == 4 else torch.int8, device=device
        )
        return {
            "data": empty_data,
            "scales": torch.empty(0, dtype=torch.float32, device=device),
            "shape": shape,
            "pad_len": 0,
            "nbits": nbits,
            "orig_dtype": orig_dtype,
            "is_quantized": True,
        }

    pad_len = (block_size - (numel % block_size)) % block_size
    if pad_len > 0:
        flat = torch.nn.functional.pad(flat, (0, pad_len))

    blocks = flat.reshape(-1, block_size)
    max_vals = blocks.abs().max(dim=-1, keepdim=True).values
    zero_mask = max_vals == 0

    if nbits == 4:
        scales = torch.where(zero_mask, torch.ones_like(max_vals), max_vals / 7.0)
        q = torch.where(
            zero_mask,
            torch.zeros_like(blocks, dtype=torch.int8),
            torch.clamp(torch.round(blocks / scales), -7, 7).to(torch.int8),
        )
        scales = torch.where(zero_mask, torch.zeros_like(max_vals), scales)

        q_flat = q.flatten()
        q_unsigned = (q_flat + 7).to(torch.uint8)
        lo = q_unsigned[0::2]
        hi = q_unsigned[1::2]
        packed = (hi << 4) | (lo & 0x0F)

        return {
            "data": packed,
            "scales": scales.squeeze(-1).to(torch.float32),
            "shape": shape,
            "pad_len": pad_len,
            "nbits": 4,
            "orig_dtype": orig_dtype,
            "is_quantized": True,
        }
    else:  # nbits == 8
        scales = torch.where(zero_mask, torch.ones_like(max_vals), max_vals / 127.0)
        q = torch.where(
            zero_mask,
            torch.zeros_like(blocks, dtype=torch.int8),
            torch.clamp(torch.round(blocks / scales), -127, 127).to(torch.int8),
        )
        scales = torch.where(zero_mask, torch.zeros_like(max_vals), scales)

        return {
            "data": q.flatten(),
            "scales": scales.squeeze(-1).to(torch.float32),
            "shape": shape,
            "pad_len": pad_len,
            "nbits": 8,
            "orig_dtype": orig_dtype,
            "is_quantized": True,
        }


def dequantize_tensor(
    quantized_dict: dict[
        str, torch.Tensor | tuple[int, ...] | int | torch.dtype | bool
    ],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Reconstruct floating-point PyTorch tensor from a quantized dictionary.

    Args:
        quantized_dict: Dictionary returned by `quantize_tensor`.
        device: Destination device for reconstructed tensor (defaults to data.device).

    Returns:
        Reconstructed PyTorch tensor matching original shape and dtype.
    """
    data = quantized_dict["data"]
    scales = quantized_dict["scales"]
    shape = quantized_dict["shape"]
    pad_len = quantized_dict["pad_len"]
    nbits = quantized_dict["nbits"]
    orig_dtype = quantized_dict.get("orig_dtype", torch.float32)
    target_device = device if device is not None else data.device

    if data.numel() == 0:
        return torch.empty(shape, dtype=orig_dtype, device=target_device)

    num_blocks = scales.numel()

    if nbits == 4:
        lo = (data & 0x0F).to(torch.float32) - 7.0
        hi = ((data >> 4) & 0x0F).to(torch.float32) - 7.0
        unpacked = torch.stack([lo, hi], dim=-1).flatten()
        block_size = unpacked.numel() // num_blocks
        blocks = unpacked.reshape(num_blocks, block_size)
    elif nbits == 8:
        block_size = data.numel() // num_blocks
        blocks = data.reshape(num_blocks, block_size).to(torch.float32)
    else:
        raise ValueError(f"Unsupported nbits: {nbits}")

    dequant = blocks * scales.unsqueeze(-1)
    flat = dequant.flatten()
    if pad_len > 0:
        flat = flat[:-pad_len]
    return flat.reshape(shape).to(dtype=orig_dtype, device=target_device)


def quantize_state_dict(
    state_dict: nn.Module | dict[str, torch.Tensor],
    nbits: int = 4,
    block_size: int = 64,
    residuals: dict[str, torch.Tensor] | None = None,
) -> OrderedDict:
    """Quantize model weights for bandwidth-efficient transmission in federated rounds.

    Args:
        state_dict: nn.Module or model parameter dictionary.
        nbits: Quantization bit-width (4 or 8).
        block_size: Elements per quantization block.
        residuals: Optional dictionary accumulating error residuals across rounds.

    Returns:
        OrderedDict mapping parameter names to quantized descriptors.
    """
    dict_to_process = (
        OrderedDict(state_dict.state_dict())
        if isinstance(state_dict, nn.Module)
        else OrderedDict(state_dict)
    )

    quantized_dict = OrderedDict()
    for k, v in dict_to_process.items():
        if isinstance(v, torch.Tensor) and torch.is_floating_point(v):
            if residuals is not None:
                res = residuals.get(k, torch.zeros_like(v))
                v_adj = v + res
                q = quantize_tensor(v_adj, nbits=nbits, block_size=block_size)
                v_recon = dequantize_tensor(q, device=v.device)
                residuals[k] = v_adj - v_recon
                quantized_dict[k] = q
            else:
                quantized_dict[k] = quantize_tensor(
                    v, nbits=nbits, block_size=block_size
                )
        elif isinstance(v, torch.Tensor):
            quantized_dict[k] = {
                "data": v.clone(),
                "is_quantized": False,
                "nbits": None,
            }
        else:
            quantized_dict[k] = v
    return quantized_dict


def dequantize_state_dict(
    quantized_dict: OrderedDict,
    target_model: nn.Module | None = None,
    device: torch.device | None = None,
) -> OrderedDict:
    """Restore state dict from quantized representation, optionally updating target model.

    Args:
        quantized_dict: OrderedDict returned by `quantize_state_dict`.
        target_model: Optional PyTorch Module to load restored parameters into.
        device: Destination device for restored tensors.

    Returns:
        OrderedDict of reconstructed float tensors.
    """
    restored = OrderedDict()
    for k, v in quantized_dict.items():
        if isinstance(v, dict) and v.get("is_quantized", False):
            restored[k] = dequantize_tensor(v, device=device)
        elif isinstance(v, dict) and "data" in v:
            t = v["data"]
            restored[k] = (
                t.to(device)
                if device is not None and isinstance(t, torch.Tensor)
                else t
            )
        elif isinstance(v, torch.Tensor):
            restored[k] = v.to(device) if device is not None else v
        else:
            restored[k] = v

    if target_model is not None:
        target_model.load_state_dict(restored, strict=False)
    return restored


def quantize_representations(
    reps: torch.Tensor,
    nbits: int = 4,
    block_size: int = 32,
) -> dict[str, torch.Tensor | tuple[int, ...] | int | torch.dtype | bool]:
    """Quantize projection representations for memory-efficient contrastive caching in MOON.

    In Model-Contrastive Federated Learning (MOON), clients cache representation vectors
    from previous local models and the global model to compute contrastive losses.
    Quantizing cached projections to INT4 dramatically reduces GPU memory overhead.

    Args:
        reps: Latent projection representation tensor (batch_size, rep_dim).
        nbits: Quantization bit-width (4 or 8).
        block_size: Elements per block.

    Returns:
        Quantized dictionary suitable for compact storage or communication.
    """
    return quantize_tensor(reps, nbits=nbits, block_size=block_size)


def dequantize_representations(
    quantized_dict: dict[
        str, torch.Tensor | tuple[int, ...] | int | torch.dtype | bool
    ],
    device: torch.device | None = None,
) -> torch.Tensor:
    """Restore representation tensor from quantized cache dictionary."""
    return dequantize_tensor(quantized_dict, device=device)


def simulate_quantization(
    model: nn.Module,
    nbits: int = 4,
    block_size: int = 64,
    in_place: bool = False,
) -> nn.Module:
    """In-memory simulation of quantization distortion on model weights."""
    target = model if in_place else copy.deepcopy(model)
    q_dict = quantize_state_dict(target, nbits=nbits, block_size=block_size)
    dequantize_state_dict(q_dict, target_model=target)
    return target


def calc_compression_stats(
    raw_data: nn.Module | dict[str, torch.Tensor] | torch.Tensor,
    quantized_data: OrderedDict | dict,
) -> dict[str, int | float]:
    """Calculate payload volume and compression metrics."""
    raw_bytes = 0
    if isinstance(raw_data, nn.Module):
        for p in raw_data.parameters():
            raw_bytes += p.numel() * p.element_size()
        for b in raw_data.buffers():
            raw_bytes += b.numel() * b.element_size()
    elif isinstance(raw_data, dict):
        for v in raw_data.values():
            if isinstance(v, torch.Tensor):
                raw_bytes += v.numel() * v.element_size()
    elif isinstance(raw_data, torch.Tensor):
        raw_bytes = raw_data.numel() * raw_data.element_size()

    compressed_bytes = 0
    if "data" in quantized_data and "scales" in quantized_data:
        compressed_bytes += (
            quantized_data["data"].numel() * quantized_data["data"].element_size()
            + quantized_data["scales"].numel() * quantized_data["scales"].element_size()
        )
    elif isinstance(quantized_data, dict):
        for v in quantized_data.values():
            if isinstance(v, dict) and v.get("is_quantized", False):
                compressed_bytes += (
                    v["data"].numel() * v["data"].element_size()
                    + v["scales"].numel() * v["scales"].element_size()
                )
            elif (
                isinstance(v, dict)
                and "data" in v
                and isinstance(v["data"], torch.Tensor)
            ):
                compressed_bytes += v["data"].numel() * v["data"].element_size()
            elif isinstance(v, torch.Tensor):
                compressed_bytes += v.numel() * v.element_size()

    ratio = raw_bytes / compressed_bytes if compressed_bytes > 0 else 1.0
    savings = (1.0 - compressed_bytes / raw_bytes) * 100.0 if raw_bytes > 0 else 0.0
    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": ratio,
        "bandwidth_savings_pct": savings,
    }
