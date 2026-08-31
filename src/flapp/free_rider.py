"""Simulates dishonest/low-effort clients by corrupting their fit() result
AFTER real local training, right before it would be sent to the server —
the same point real free-riding would occur.

Three behaviors:
  silent: sends an all-zero update (contributes nothing, costs nothing)
  noisy:  sends the update plus large Gaussian noise (garbage, not zero)
  lazy:   sends a scaled-down update (as if it trained much less than it claims)
"""
import numpy as np

BEHAVIORS = ("silent", "noisy", "lazy")
_LAZY_SCALE = 0.3  # a lazy client's update counts for ~30% of an honest one's
_NOISE_MULTIPLIER = 3.0  # noise std relative to each layer's OWN std (see note below)


def assign_behavior(partition_id: int, num_free_riders: int) -> str:
    """Deterministic: the first `num_free_riders` partition ids cycle
    through BEHAVIORS; every other partition id is honest."""
    if partition_id >= num_free_riders:
        return "honest"
    return BEHAVIORS[partition_id % len(BEHAVIORS)]


def apply_behavior(
    params: list[np.ndarray], behavior: str, rng: np.random.Generator | None = None
) -> list[np.ndarray]:
    if behavior == "honest":
        return params
    if behavior == "silent":
        return [np.zeros_like(p) for p in params]
    if behavior == "noisy":
        # Noise scale is relative to each layer's OWN std, not a fixed
        # absolute constant: real model weights range from ~0.01 std
        # (dense layers) to ~0.2 std (conv layers) in this architecture,
        # so a fixed absolute noise scale would either be negligible on
        # large-magnitude layers or (as originally found during full-scale
        # validation) completely swamp small-magnitude layers, turning
        # "noisy" into "replaced with pure random noise" and collapsing
        # FedAvg to near-random accuracy — not a useful free-rider scenario
        # to compare strategies against.
        rng = rng if rng is not None else np.random.default_rng()
        result = []
        for p in params:
            layer_std = np.std(p)
            noise_std = layer_std * _NOISE_MULTIPLIER if layer_std > 0 else 0.1
            result.append(p + rng.normal(scale=noise_std, size=p.shape).astype(p.dtype))
        return result
    if behavior == "lazy":
        return [p * _LAZY_SCALE for p in params]
    raise ValueError(f"Unknown free-rider behavior: {behavior!r}")
