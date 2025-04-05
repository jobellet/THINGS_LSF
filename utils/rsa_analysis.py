# utils/rsa_analysis.py
"""
rsa_analysis.py

This module provides functions for performing Representational Similarity Analysis (RSA)
using JAX. It includes:

  - JAX-accelerated functions for computing Spearman rank correlations.
  - Pairwise distance functions (for Euclidean and correlation distances) on fixed batches.
"""

import os
import numpy as np
from scipy.stats import rankdata
import jax
import jax.numpy as jnp
from jax import jit
from tqdm import tqdm

# --- Constants and precomputed indices for fixed batch size ---
BATCH_SIZE = 20
I_UPPER, J_UPPER = np.triu_indices(BATCH_SIZE, k=1)
I_UPPER = jnp.array(I_UPPER)
J_UPPER = jnp.array(J_UPPER)
nRDMfeatures = len(I_UPPER)

####################################
# JAX functions for RSA
####################################
@jit
def compute_spearman_rankcorr(x_ranked, y_ranked):
    """
    Compute Spearman's rank correlation between two rank-transformed vectors
    using Pearson's formula.
    """
    x_mean = jnp.mean(x_ranked)
    y_mean = jnp.mean(y_ranked)
    num = jnp.sum((x_ranked - x_mean) * (y_ranked - y_mean))
    den = jnp.sqrt(jnp.sum((x_ranked - x_mean)**2) * jnp.sum((y_ranked - y_mean)**2))
    return num / den

@jit
def compute_rsa_jax(rdm1_ranked, rdm2_ranked):
    """Wrapper to compute RSA (Spearman correlation) on rank-transformed RDMs."""
    return compute_spearman_rankcorr(rdm1_ranked, rdm2_ranked)

####################################
# Pairwise distance functions (fixed batch size)
####################################
@jit
def pairwise_euclidean_distance_fixed(X):
    """
    Compute the upper-triangular Euclidean distances (flattened) for a batch of data.
    
    Parameters
    ----------
    X : array of shape [BATCH_SIZE, feature_dim]
    
    Returns
    -------
    distances : array of shape [nRDMfeatures,]
    """
    diff = X[:, None, :] - X[None, :, :]
    sq = jnp.sum(diff**2, axis=-1)
    return sq[I_UPPER, J_UPPER]

@jit
def pairwise_correlation_distance_fixed(X):
    """
    Compute the upper-triangular correlation distances (flattened) for a batch of data.
    
    Parameters
    ----------
    X : array of shape [BATCH_SIZE, feature_dim]
    
    Returns
    -------
    distances : array of shape [nRDMfeatures,]
    """
    X_mean = jnp.mean(X, axis=1, keepdims=True)
    X_centered = X - X_mean
    norms = jnp.sqrt(jnp.sum(X_centered**2, axis=1, keepdims=True))
    eps = 1e-8
    X_normalized = X_centered / (norms + eps)
    corr_matrix = X_normalized @ X_normalized.T
    dist_matrix = 1 - corr_matrix
    return dist_matrix[I_UPPER, J_UPPER]


