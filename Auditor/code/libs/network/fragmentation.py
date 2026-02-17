# connectedness_entropy.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class NormEntropyResult:
    n: int
    n_components: int
    component_sizes: np.ndarray          # descending
    norm_entropy: float                  # in [0,1]
    n_edges_rows: int                    # rows after filtering (may include duplicates / both directions)
    n_edges_undirected_unique: int       # unique undirected edges among recommended nodes


def _unique_undirected_pairs(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Return unique undirected pairs as an array of shape (E,2) with rows (min, max).
    """
    pairs = np.stack([u, v], axis=1).astype(np.int64, copy=False)
    pairs = np.sort(pairs, axis=1)               # (min, max)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]    # drop self-loops
    pairs = np.unique(pairs, axis=0)             # drop duplicates
    return pairs


def norm_entropy_from_component_sizes(component_sizes: np.ndarray) -> float:
    """
    Normalized entropy of component size distribution, in [0,1].
    """
    sizes = np.asarray(component_sizes, dtype=np.float64)
    n = sizes.sum()
    if n <= 1:
        return 0.0
    p = sizes / n
    H = -np.sum(p * np.log(p))
    return float(H / np.log(n))


def norm_entropy_R_from_edgelist(
    rec_ids: Sequence[int],
    edges: pd.DataFrame,
    *,
    src_col: str = "src",
    dst_col: str = "dst",
) -> NormEntropyResult:
    """
    Compute NormEntropy(R) for a unique list of recommended author IDs.

    Parameters
    ----------
    rec_ids:
        Unique recommended node ids (ints). Order does not matter.
    edges:
        Edge list DataFrame for the full APS coauthorship graph with columns [src_col, dst_col].
        The file may contain duplicates and/or both directions; this function deduplicates
        within the induced subgraph.

    Returns
    -------
    NormEntropyResult with diagnostics and norm_entropy in [0,1].
    """
    rec_ids = np.asarray(rec_ids, dtype=np.int64)
    n = int(rec_ids.size)

    if n <= 1:
        return NormEntropyResult(
            n=n,
            n_components=n,
            component_sizes=np.array([] if n == 0 else [n], dtype=int),
            norm_entropy=None,
            n_edges_rows=None,
            n_edges_undirected_unique=None,
        )

    # Membership test for filtering edges: O(|E|) scan.
    rec_set = set(map(int, rec_ids))

    src = edges[src_col].to_numpy(dtype=np.int64, copy=False)
    dst = edges[dst_col].to_numpy(dtype=np.int64, copy=False)

    mask = np.fromiter(((int(u) in rec_set) and (int(v) in rec_set) for u, v in zip(src, dst)),
                       count=src.size, dtype=bool)
    src_f = src[mask]
    dst_f = dst[mask]

    n_edges_rows = int(src_f.size)

    # Deduplicate to unique undirected edges among R
    pairs = _unique_undirected_pairs(src_f, dst_f)
    n_edges_undirected_unique = int(pairs.shape[0])

    # Map rec_ids to 0..n-1 indices for a small induced adjacency
    # (rec_ids are unique, per your setup)
    idx: Dict[int, int] = {int(a): i for i, a in enumerate(rec_ids)}

    if n_edges_undirected_unique == 0:
        # No edges: all isolated components
        comp_sizes = np.ones(n, dtype=int)
        ne = norm_entropy_from_component_sizes(comp_sizes)
        return NormEntropyResult(
            n=n,
            n_components=n,
            component_sizes=comp_sizes,   # already all ones
            norm_entropy=ne,
            n_edges_rows=n_edges_rows,
            n_edges_undirected_unique=0,
        )

    # Convert unique undirected pairs (ids) into induced indices
    u_idx = np.fromiter((idx[int(u)] for u in pairs[:, 0]), count=n_edges_undirected_unique, dtype=np.int64)
    v_idx = np.fromiter((idx[int(v)] for v in pairs[:, 1]), count=n_edges_undirected_unique, dtype=np.int64)

    # Build symmetric adjacency
    row = np.concatenate([u_idx, v_idx])
    col = np.concatenate([v_idx, u_idx])
    data = np.ones(row.size, dtype=np.uint8)

    A = sp.coo_matrix((data, (row, col)), shape=(n, n)).tocsr()
    A.data[:] = 1
    A.eliminate_zeros()

    n_comp, labels = connected_components(A, directed=False, connection="weak")
    comp_sizes = np.bincount(labels, minlength=n_comp)
    comp_sizes = np.sort(comp_sizes)[::-1]

    ne = norm_entropy_from_component_sizes(comp_sizes)

    return NormEntropyResult(
        n=n,
        n_components=int(n_comp),
        component_sizes=comp_sizes.astype(int),
        norm_entropy=float(ne),
        n_edges_rows=n_edges_rows,
        n_edges_undirected_unique=n_edges_undirected_unique,
    )


def load_edges(
    path: str,
    *,
    fmt: str = "parquet",
    src_col: str = "src",
    dst_col: str = "dst",
    sep: str = "\t",
) -> pd.DataFrame:
    """
    Convenience loader for edge lists.

    fmt:
      - "parquet": expects columns src_col, dst_col
      - "csv": expects header with src_col, dst_col
      - "tsv": expects header with src_col, dst_col
    """
    if fmt == "parquet":
        df = pd.read_parquet(path, columns=[src_col, dst_col])
    elif fmt == "csv":
        df = pd.read_csv(path, usecols=[src_col, dst_col])
    elif fmt == "tsv":
        df = pd.read_csv(path, sep=sep, usecols=[src_col, dst_col])
    else:
        raise ValueError(f"Unknown fmt={fmt!r}")
    # enforce integer dtype
    df[src_col] = df[src_col].astype(np.int64, copy=False)
    df[dst_col] = df[dst_col].astype(np.int64, copy=False)
    return df
