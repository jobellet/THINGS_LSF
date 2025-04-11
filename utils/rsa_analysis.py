"""
rsa_analysis.py

This module provides functions for performing Representational Similarity Analysis (RSA)
using JAX. It includes JAX-accelerated functions for computing Spearman rank correlations
and pairwise distance functions (for Euclidean and correlation distances) that work on a
user-chosen batch size.
"""

import numpy as np
from scipy.stats import rankdata
import jax
import jax.numpy as jnp
from jax import jit
from functools import lru_cache
from tqdm import tqdm
from functools import partial

@partial(jit, static_argnums=(1,))
def pairwise_euclidean_distance_fixed(X, batch_size):
    i_upper, j_upper = _compute_indices(batch_size)
    diff = X[:, None, :] - X[None, :, :]
    sq = jnp.sum(diff ** 2, axis=-1)
    return sq[i_upper, j_upper]

@partial(jit, static_argnums=(1,))
def pairwise_correlation_distance_fixed(X, batch_size):
    i_upper, j_upper = _compute_indices(batch_size)
    X_mean = jnp.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean
    norms = jnp.sqrt(jnp.sum(X_centered ** 2, axis=1, keepdims=True))
    eps = 1e-8
    X_normalized = X_centered / (norms + eps)
    corr_matrix = X_normalized @ X_normalized.T
    dist_matrix = 1 - corr_matrix
    return dist_matrix[i_upper, j_upper]

@lru_cache(maxsize=None)
def _compute_indices(batch_size: int):
    """
    Compute the upper-triangular indices for a square matrix of size `batch_size`
    and cache the result for each unique batch size.

    Parameters
    ----------
    batch_size : int
        Number of items in the batch (rows/columns).

    Returns
    -------
    i_upper : jnp.array
        Row indices of the upper triangular (excluding diagonal).
    j_upper : jnp.array
        Column indices of the upper triangular (excluding diagonal).
    """
    I_UPPER, J_UPPER = np.triu_indices(batch_size, k=1)
    return jnp.array(I_UPPER), jnp.array(J_UPPER)

@jit
def compute_spearman_rankcorr(x_ranked: jnp.ndarray, y_ranked: jnp.ndarray) -> jnp.ndarray:
    """
    Compute Spearman's rank correlation between two rank-transformed vectors
    using Pearson's formula.

    Parameters
    ----------
    x_ranked : jnp.ndarray
        Rank-transformed vector (e.g., from rankdata).
    y_ranked : jnp.ndarray
        Rank-transformed vector (e.g., from rankdata).

    Returns
    -------
    corr : jnp.ndarray
        Scalar representing Spearman correlation between x_ranked and y_ranked.
    """
    x_mean = jnp.mean(x_ranked)
    y_mean = jnp.mean(y_ranked)
    num = jnp.sum((x_ranked - x_mean) * (y_ranked - y_mean))
    den = jnp.sqrt(jnp.sum((x_ranked - x_mean) ** 2) * jnp.sum((y_ranked - y_mean) ** 2))
    return num / (den + 1e-12)  # add a small epsilon to avoid division by zero

@jit
def compute_rsa_jax(rdm1_ranked: jnp.ndarray, rdm2_ranked: jnp.ndarray) -> jnp.ndarray:
    """
    Wrapper to compute RSA (Spearman correlation) on rank-transformed RDMs.

    Parameters
    ----------
    rdm1_ranked : jnp.ndarray
        Flattened rank-transformed RDM.
    rdm2_ranked : jnp.ndarray
        Flattened rank-transformed RDM.

    Returns
    -------
    corr : jnp.ndarray
        Spearman correlation between the two RDMs.
    """
    return compute_spearman_rankcorr(rdm1_ranked, rdm2_ranked)