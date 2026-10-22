"""Differentiable RGB block DCT and frequency-constrained identity PGD."""

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F


def dct_matrix(size, reference):
    frequencies = torch.arange(size, device=reference.device, dtype=reference.dtype)
    matrix = torch.cos(math.pi / size * frequencies[:, None] * (frequencies + 0.5))
    matrix *= math.sqrt(2.0 / size)
    matrix[0] /= math.sqrt(2.0)
    return matrix


def block_dct(image, block_size=8):
    """NCHW [0, 1] RGB -> (N, C, block_rows, block_cols, u, v), orthonormal."""
    if image.ndim != 4 or not image.is_floating_point():
        raise ValueError("image must be a floating-point NCHW tensor")
    if block_size < 1 or any(size % block_size for size in image.shape[-2:]):
        raise ValueError("image dimensions must be divisible by block_size")
    blocks = image.unfold(2, block_size, block_size).unfold(3, block_size, block_size)
    basis = dct_matrix(block_size, image)
    return basis @ blocks @ basis.T


def block_idct(coefficients):
    """Inverse of block_dct; the input gradient is preserved."""
    if coefficients.ndim != 6 or coefficients.shape[-2] != coefficients.shape[-1]:
        raise ValueError("coefficients must have shape (N, C, rows, cols, B, B)")
    n, channels, rows, cols, size, _ = coefficients.shape
    basis = dct_matrix(size, coefficients)
    blocks = basis.T @ coefficients @ basis
    return blocks.permute(0, 1, 2, 4, 3, 5).reshape(n, channels, rows * size, cols * size)


@dataclass(frozen=True)
class AttackConfig:
    steps: int = 20
    step_size: float = 1 / 255
    epsilon: float = 8 / 255
    mid_budget: float = 0.08
    high_budget: float = 0.04
    bands: str = "high"
    seed: int = 0
    init_scale: float = 0.001

    def __post_init__(self):
        if not isinstance(self.steps, int) or self.steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        for name in ("step_size", "epsilon", "mid_budget", "high_budget", "init_scale"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.bands not in {"high", "mid-high"}:
            raise ValueError("bands must be high or mid-high")


def frequency_budget(config, reference):
    coords = torch.arange(8, device=reference.device)
    frequency = coords[:, None] + coords[None, :]
    budget = torch.zeros((8, 8), dtype=reference.dtype, device=reference.device)
    if config.bands == "mid-high":
        budget[(frequency >= 3) & (frequency <= 6)] = config.mid_budget
    budget[frequency >= 7] = config.high_budget
    return budget


def project_delta(delta, mask, budget, epsilon):
    """Preserve exact coefficient support while bounding masked pixel changes."""
    delta = torch.maximum(torch.minimum(delta, budget), -budget)
    magnitude = (mask * block_idct(delta)).abs().flatten(1).amax(1)
    scale = (epsilon / magnitude.clamp_min(torch.finfo(delta.dtype).tiny)).clamp(max=1)
    return delta * scale.reshape(-1, 1, 1, 1, 1, 1)


@dataclass
class AttackResult:
    image: torch.Tensor
    delta: torch.Tensor
    history: list
    cosine_before: float
    cosine_after: float


def protect_face(image, mask, recognizer, config=AttackConfig()):
    """Optimize one aligned face. recognizer accepts RGB [0, 1] and returns embeddings."""
    if image.ndim != 4 or image.shape[0:2] != (1, 3):
        raise ValueError("image must have shape (1, 3, H, W)")
    if mask.shape != (1, 1, *image.shape[-2:]):
        raise ValueError("mask must have shape (1, 1, H, W)")
    if image.device != mask.device or image.dtype != mask.dtype:
        raise ValueError("image and mask must have the same device and dtype")
    for name, tensor in (("image", image), ("mask", mask)):
        if not torch.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
            raise ValueError(f"{name} must be finite and in [0, 1]")
    image, mask = image.detach(), mask.detach()
    coefficients = block_dct(image)
    budget = frequency_budget(config, image)
    recognizer.eval()
    recognizer.requires_grad_(False)
    with torch.no_grad():
        original_embedding = F.normalize(recognizer(image), dim=1)
        cosine_before = float((original_embedding * original_embedding).sum())

    delta = torch.zeros_like(coefficients)
    best_delta = delta.clone()
    best_score = cosine_before
    history = [{"step": 0, "cosine": cosine_before, "pixel_linf": 0.0}]
    if not mask.any() or config.steps == 0 or not budget.any() or config.epsilon == 0:
        return AttackResult(image.clone(), best_delta, history, cosine_before, best_score)

    # Identical normalized embeddings are a stationary point of cosine similarity.
    generator = torch.Generator(device=image.device).manual_seed(config.seed)
    delta.uniform_(-config.init_scale, config.init_scale, generator=generator)
    delta = project_delta(delta, mask, budget, config.epsilon)
    for step in range(config.steps + 1):
        delta = delta.detach().requires_grad_(step < config.steps)
        with torch.set_grad_enabled(step < config.steps):
            candidate = (image + mask * block_idct(delta)).clamp(0, 1)
            embedding = F.normalize(recognizer(candidate), dim=1)
            loss = (embedding * original_embedding).sum()
        if not torch.isfinite(loss):
            raise RuntimeError("recognizer returned a non-finite identity loss")
        score = float(loss.detach())
        history.append({
            "step": step,
            "phase": "initialization" if step == 0 else "update",
            "cosine": score,
            "pixel_linf": float((candidate.detach() - image).abs().max()),
        })
        if score < best_score:
            best_score = score
            best_delta = delta.detach().clone()
        if step < config.steps:
            gradient, = torch.autograd.grad(loss, delta)
            if not torch.isfinite(gradient).all():
                raise RuntimeError("non-finite frequency gradient")
            with torch.no_grad():
                delta = project_delta(
                    delta - config.step_size * gradient.sign(), mask, budget, config.epsilon
                )
    with torch.no_grad():
        protected = (image + mask * block_idct(best_delta)).clamp(0, 1)
    return AttackResult(protected, best_delta, history, cosine_before, best_score)
