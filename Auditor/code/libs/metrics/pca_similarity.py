"""
pca_similarity.py

Scalable PCA-based similarity for large N.
- Handles missing values (median), optional log1p for nonnegative heavy-tailed metrics,
  standardizes metrics (StandardScaler), then PCA.
- Supports similarity for a single pair without allocating any NxN matrix.
- Supports top-k neighbor queries (recommended instead of full similarity matrix).
- Can be saved/loaded with joblib for use in another module.

Key rule for your scale (N ~ 481k):
NEVER compute a full pairwise distance/similarity matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Union, List

import json
import h5py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, normalize


ScientistID = Union[str, int]


class Log1pNonNegative(BaseEstimator, TransformerMixin):
    """
    Applies log1p to nonnegative values. Negatives are clipped to 0.
    Good default for counts like citations/publications.
    """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X = np.clip(X, a_min=0.0, a_max=None)
        return np.log1p(X)


def estimate_sigma_by_sampling(
    Z: np.ndarray,
    n_pairs: int = 200_000,
    random_state: int = 0,
) -> float:
    """
    Estimate a reasonable RBF sigma using random pairs.
    This is O(n_pairs * k) and does NOT allocate an NxN matrix.
    """
    rng = np.random.default_rng(random_state)
    n = Z.shape[0]
    if n < 2:
        return 1.0

    i = rng.integers(0, n, size=n_pairs)
    j = rng.integers(0, n, size=n_pairs)

    mask = i != j
    i, j = i[mask], j[mask]
    if i.size == 0:
        return 1.0

    diffs = Z[i] - Z[j]
    d = np.sqrt(np.sum(diffs * diffs, axis=1))
    d = d[d > 0]
    sigma = float(np.median(d)) if d.size else 1.0
    return max(sigma, 1e-12)


def rbf_similarity_from_vectors(za: np.ndarray, zb: np.ndarray, sigma: float) -> float:
    """
    RBF similarity exp(-||a-b||^2 / (2*sigma^2)).
    """
    diff = za - zb
    d2 = float(np.dot(diff, diff))
    sigma = max(float(sigma), 1e-12)
    return float(np.exp(-d2 / (2.0 * sigma * sigma)))


@dataclass
class PCASimilarityModel:
    pipeline: Pipeline
    ids: np.ndarray
    embedding_: np.ndarray
    feature_cols: Tuple[str, ...]
    n_components_: int
    explained_variance_ratio_: np.ndarray

    # Only used for RBF similarity:
    rbf_sigma_: Optional[float] = None

    # If True, embedding_ has been L2-normalized (good for cosine)
    embedding_l2_normalized_: bool = True

    # -------------------------
    # HDF5 SAVE
    # -------------------------
    def save_h5(self, path: str) -> None:
        path = Path(path)

        with h5py.File(path, "w") as f:
            # arrays
            f.create_dataset("ids", data=self.ids)
            f.create_dataset("embedding_", data=self.embedding_)
            f.create_dataset("explained_variance_ratio_", data=self.explained_variance_ratio_)

            # PCA internals
            pca = self.pipeline.named_steps["pca"]
            f.create_dataset("pca/components_", data=pca.components_)
            f.create_dataset("pca/mean_", data=pca.mean_)

            # metadata
            meta = {
                "feature_cols": list(self.feature_cols),
                "n_components_": int(self.n_components_),
                "rbf_sigma_": self.rbf_sigma_,
                "embedding_l2_normalized_": self.embedding_l2_normalized_,
            }

            f.attrs["meta_json"] = json.dumps(meta)

    # -------------------------
    # HDF5 LOAD
    # -------------------------
    @classmethod
    def load_h5(cls, path: str) -> "PCASimilarityModel":
        path = Path(path)

        with h5py.File(path, "r") as f:
            ids = f["ids"][()]
            embedding_ = f["embedding_"][()]
            evr = f["explained_variance_ratio_"][()]
            components = f["pca/components_"][()]
            mean = f["pca/mean_"][()]
            meta = json.loads(f.attrs["meta_json"])

        # rebuild PCA
        pca = PCA(n_components=meta["n_components_"])
        pca.components_ = components
        pca.mean_ = mean
        pca.explained_variance_ratio_ = evr

        pipeline = Pipeline([("pca", pca)])

        return cls(
            pipeline=pipeline,
            ids=ids,
            embedding_=embedding_,
            feature_cols=tuple(meta["feature_cols"]),
            n_components_=meta["n_components_"],
            explained_variance_ratio_=evr,
            rbf_sigma_=meta["rbf_sigma_"],
            embedding_l2_normalized_=meta["embedding_l2_normalized_"],
        )

    # def save(self, path: str) -> None:
    #     joblib.dump(self, path)

    # @staticmethod
    # def load(path: str) -> "PCASimilarityModel":
    #     return joblib.load(path)

    def index_of(self, scientist_id: ScientistID) -> int:
        matches = np.where(self.ids == scientist_id)[0]
        if len(matches) != 1:
            raise KeyError(f"Scientist id '{scientist_id}' not found uniquely.")
        return int(matches[0])

    def cosine_sim(self, a_id: ScientistID, b_id: ScientistID) -> float:
        """
        Cosine similarity in PCA space.
        Requires embedding_l2_normalized_=True for clean interpretation,
        but works either way.
        """
        ia, ib = self.index_of(a_id), self.index_of(b_id)
        za = self.embedding_[ia:ia + 1]
        zb = self.embedding_[ib:ib + 1]
        return float(cosine_similarity(za, zb)[0, 0])

    def rbf_sim(self, a_id: ScientistID, b_id: ScientistID, sigma: Optional[float] = None) -> float:
        """
        RBF similarity in PCA space. Does NOT allocate any NxN matrix.
        If sigma is None, uses self.rbf_sigma_ (must exist).
        """
        if sigma is None:
            if self.rbf_sigma_ is None:
                raise ValueError(
                    "rbf_sigma_ is not set. Fit with compute_rbf_sigma=True "
                    "or provide sigma explicitly."
                )
            sigma = self.rbf_sigma_

        ia, ib = self.index_of(a_id), self.index_of(b_id)
        za = self.embedding_[ia]
        zb = self.embedding_[ib]
        return rbf_similarity_from_vectors(za, zb, sigma=float(sigma))

    def embed_dataframe(self, df: pd.DataFrame, l2_normalize_output: Optional[bool] = None) -> np.ndarray:
        """
        Embed rows of df into PCA space using the stored preprocessing + PCA.

        l2_normalize_output:
        - If None, follows how this model stores embedding_ (embedding_l2_normalized_).
        - If True, L2-normalize rows (recommended for cosine similarity).
        - If False, keep raw PCA scores (recommended for Euclidean/RBF geometry).
        """
        if l2_normalize_output is None:
            l2_normalize_output = self.embedding_l2_normalized_

        X_pre = self.pipeline.named_steps["preprocess"].transform(df)
        Z = self.pipeline.named_steps["pca"].transform(X_pre)
        return normalize(Z, norm="l2") if l2_normalize_output else Z

    def topk_similar(
        self,
        query_id: ScientistID,
        k: int = 20,
        method: str = "cosine",
    ) -> List[Tuple[ScientistID, float]]:
        """
        Return top-k most similar scientists to query_id without NxN matrices.

        method:
        - "cosine": uses cosine similarity (best with L2-normalized embedding_)
        - "rbf": uses Euclidean neighbors then converts distance to RBF similarity
        """
        method = method.lower().strip()
        iq = self.index_of(query_id)

        if method == "cosine":
            # Use NearestNeighbors with cosine distance; similarity = 1 - distance
            nbrs = NearestNeighbors(
                n_neighbors=min(k + 1, self.embedding_.shape[0]),
                metric="cosine",
                algorithm="auto",
            )
            nbrs.fit(self.embedding_)
            dists, idxs = nbrs.kneighbors(self.embedding_[iq:iq + 1], return_distance=True)
            dists = dists[0]
            idxs = idxs[0]
            out = []
            for dist, idx in zip(dists, idxs):
                if idx == iq:
                    continue
                out.append((self.ids[idx], float(1.0 - dist)))
                if len(out) >= k:
                    break
            return out

        if method == "rbf":
            if self.rbf_sigma_ is None:
                raise ValueError("rbf_sigma_ is not set; fit with compute_rbf_sigma=True.")

            # For RBF, neighbor search should be in Euclidean space.
            # This is more meaningful when embedding is NOT L2-normalized.
            nbrs = NearestNeighbors(
                n_neighbors=min(k + 1, self.embedding_.shape[0]),
                metric="euclidean",
                algorithm="auto",
            )
            nbrs.fit(self.embedding_)
            dists, idxs = nbrs.kneighbors(self.embedding_[iq:iq + 1], return_distance=True)
            dists = dists[0]
            idxs = idxs[0]

            out = []
            sigma = float(self.rbf_sigma_)
            for dist, idx in zip(dists, idxs):
                if idx == iq:
                    continue
                sim = float(np.exp(-(dist * dist) / (2.0 * sigma * sigma)))
                out.append((self.ids[idx], sim))
                if len(out) >= k:
                    break
            return out

        raise ValueError("method must be 'cosine' or 'rbf'.")


def fit_pca_similarity(
    df: pd.DataFrame,
    id_col: str,
    metric_cols: Optional[Iterable[str]] = None,
    explained_variance_target: float = 0.9,
    max_components: Optional[int] = None,
    use_log1p: bool = True,
    # If True, store L2-normalized embedding_ (recommended for cosine similarity)
    l2_normalize_embedding: bool = True,
    # If True, estimate and store rbf_sigma_ using sampling (no NxN)
    compute_rbf_sigma: bool = False,
    rbf_sigma_n_pairs: int = 200_000,
    random_state: int = 0,
) -> PCASimilarityModel:
    """
    Fits preprocessing + PCA and returns a reusable, savable model.

    Standardization: YES (StandardScaler is always applied).
    """

    if id_col not in df.columns:
        raise ValueError(f"id_col '{id_col}' not in dataframe columns.")

    ids = df[id_col].to_numpy()

    if metric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        metric_cols = [c for c in numeric_cols if c != id_col]
    metric_cols = tuple(metric_cols)

    if len(metric_cols) == 0:
        raise ValueError("No metric columns selected.")

    # Preprocess: impute -> (optional log1p) -> standardize
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if use_log1p:
        steps.append(("log1p", Log1pNonNegative()))
    steps.append(("scaler", StandardScaler()))

    preprocessor = ColumnTransformer(
        transformers=[("num", Pipeline(steps=steps), list(metric_cols))],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    X_pre = preprocessor.fit_transform(df)
    n_features = X_pre.shape[1]
    full_k = min(n_features, X_pre.shape[0])
    if full_k < 1:
        raise ValueError("Not enough data after preprocessing.")

    # Choose k
    pca_full = PCA(n_components=full_k, random_state=random_state)
    pca_full.fit(X_pre)
    cum = np.cumsum(pca_full.explained_variance_ratio_)

    if max_components is None:
        k = int(np.searchsorted(cum, explained_variance_target) + 1)
    else:
        k = int(max_components)
    k = max(1, min(k, full_k))

    # Fit final PCA
    pca = PCA(n_components=k, random_state=random_state)
    Z = pca.fit_transform(X_pre)

    rbf_sigma = None
    if compute_rbf_sigma:
        # IMPORTANT:
        # Estimate sigma in the geometry you will actually use for RBF.
        # If you plan to use RBF, you should set l2_normalize_embedding=False.
        rbf_sigma = estimate_sigma_by_sampling(Z, n_pairs=rbf_sigma_n_pairs, random_state=random_state)

    if l2_normalize_embedding:
        Z_store = normalize(Z, norm="l2")
    else:
        Z_store = Z

    pipeline = Pipeline([("preprocess", preprocessor), ("pca", pca)])

    return PCASimilarityModel(
        pipeline=pipeline,
        ids=ids,
        embedding_=Z_store,
        feature_cols=metric_cols,
        n_components_=k,
        explained_variance_ratio_=pca.explained_variance_ratio_.copy(),
        rbf_sigma_=rbf_sigma,
        embedding_l2_normalized_=l2_normalize_embedding,
    )
