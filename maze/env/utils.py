import os
import jax.numpy as jnp
import jax
import numpy as np

# Define project root based on environment variable or relative to this file
DEFAULT_PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

def convert_path(path: str) -> str:
    """Convert a relative path to absolute by joining with the project root."""
    if os.path.isabs(path):
        return path
    return os.path.join(DEFAULT_PROJECT_ROOT, path)

def get_tensor_stats(xs: jax.Array, mask: jax.Array, n: int):
    """Get stats about a tensor, used for logging."""
    mean = (xs * mask).sum() / n
    mask = mask.astype(jnp.bool_)
    return dict(
        mean=mean,
        min=jnp.min(xs, where=mask, initial=float('inf')),
        max=jnp.max(xs, where=mask, initial=float('-inf')),
        std=jnp.std(xs, where=mask),
    )

def get_tensor_stats_np(xs: np.ndarray, mask: np.ndarray, n: int):
    """Get stats about a tensor, used for logging."""
    mean = (xs * mask).sum() / n
    mask = mask.astype(np.bool_)
    return dict(
        mean=mean,
        min=np.min(xs, where=mask, initial=float('inf')),
        max=np.max(xs, where=mask, initial=float('-inf')),
        std=np.std(xs, where=mask),
    )

def unpad_array(xs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    pad_t = jnp.where(1 - mask)[0]
    if len(pad_t) > 0:
        xs = xs[:pad_t[0]]
    return xs
