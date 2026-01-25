import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def embed(texts, model_name="all-mpnet-base-v2", batch_size=64):
    model = SentenceTransformer(model_name)
    return model.encode(
        [str(t) for t in texts],
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    )

def build_prototypes(cluster_seeds: dict, model_name="all-mpnet-base-v2"):
    """
    cluster_seeds: dict[str, list[str]]
      e.g. {"cluster_A": ["seed sent 1", "seed sent 2"], "cluster_B": [...]}
    """
    all_seeds = []
    cluster_names = []
    for cname, seeds in cluster_seeds.items():
        for s in seeds:
            all_seeds.append(s)
            cluster_names.append(cname)

    seed_emb = embed(all_seeds, model_name=model_name)

    # average per cluster
    prototypes = {}
    for cname in cluster_seeds.keys():
        idx = [i for i, n in enumerate(cluster_names) if n == cname]
        prototypes[cname] = seed_emb[idx].mean(axis=0)

    proto_mat = np.vstack([prototypes[c] for c in prototypes.keys()])
    proto_names = list(prototypes.keys())
    return proto_names, proto_mat

def assign_to_prototypes(texts, proto_names, proto_mat, model_name="all-mpnet-base-v2", threshold=None):
    """
    threshold: if set (e.g. 0.35-0.55), items below threshold become 'unknown'
    Returns DataFrame with best cluster + score + full similarity vector.
    """
    x_emb = embed(texts, model_name=model_name)
    sims = cosine_similarity(x_emb, proto_mat)  # (n_texts, n_clusters)

    best_idx = sims.argmax(axis=1)
    best_score = sims.max(axis=1)
    best_cluster = [proto_names[i] for i in best_idx]

    if threshold is not None:
        best_cluster = [
            c if sc >= threshold else "unknown"
            for c, sc in zip(best_cluster, best_score)
        ]

    return pd.DataFrame({
        "assigned_cluster": best_cluster,
        "score": best_score,
    }), sims





# """
# text_clustering_sbert.py

# Modular utilities to cluster a pandas Series of texts using SBERT embeddings.
# Supports:
#   - KMeans (fixed k)
#   - Agglomerative clustering (unknown k; use distance threshold)
#   - HDBSCAN (unknown k; supports outliers)

# Requirements:
#   pip install -U sentence-transformers scikit-learn numpy pandas
# Optional (recommended for text):
#   pip install -U hdbscan
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List, Optional, Tuple, Literal, Union

# import numpy as np
# import pandas as pd

# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans, AgglomerativeClustering
# from sklearn.metrics import silhouette_score
# from sklearn.metrics.pairwise import cosine_similarity


# # ---------------------------
# # Configs
# # ---------------------------

# ClusteringMethod = Literal["kmeans", "agglomerative", "hdbscan"]


# @dataclass(frozen=True)
# class EmbedConfig:
#     model_name: str = "all-mpnet-base-v2"
#     batch_size: int = 64
#     normalize: bool = True
#     show_progress_bar: bool = True


# @dataclass(frozen=True)
# class KMeansConfig:
#     n_clusters: int = 20
#     random_state: int = 0
#     n_init: Union[int, str] = "auto"


# @dataclass(frozen=True)
# class AgglomerativeConfig:
#     # distance_threshold is cosine distance (1 - cosine similarity)
#     distance_threshold: float = 0.25
#     linkage: str = "average"


# @dataclass(frozen=True)
# class HDBSCANConfig:
#     min_cluster_size: int = 10
#     min_samples: int = 5
#     # Use "euclidean" with normalized embeddings; HDBSCAN supports many metrics.
#     metric: str = "euclidean"


# # ---------------------------
# # Embedding
# # ---------------------------

# def embed_texts(
#     texts: Union[pd.Series, List[str]],
#     cfg: EmbedConfig = EmbedConfig(),
# ) -> np.ndarray:
#     """
#     Encode texts into SBERT embeddings.

#     Returns:
#         emb: np.ndarray shape (n_texts, dim)
#     """
#     if isinstance(texts, pd.Series):
#         texts_list = texts.astype(str).tolist()
#     else:
#         texts_list = [str(t) for t in texts]

#     model = SentenceTransformer(cfg.model_name)
#     emb = model.encode(
#         texts_list,
#         batch_size=cfg.batch_size,
#         show_progress_bar=cfg.show_progress_bar,
#         convert_to_numpy=True,
#         normalize_embeddings=cfg.normalize,
#     )
#     return emb


# # ---------------------------
# # Clustering
# # ---------------------------

# def cluster_kmeans(emb: np.ndarray, cfg: KMeansConfig = KMeansConfig()) -> np.ndarray:
#     """
#     KMeans clustering for a fixed number of clusters.
#     With normalized embeddings, Euclidean distance aligns closely with cosine distance.
#     """
#     km = KMeans(
#         n_clusters=cfg.n_clusters,
#         n_init=cfg.n_init,
#         random_state=cfg.random_state,
#     )
#     return km.fit_predict(emb)


# def cluster_agglomerative(
#     emb: np.ndarray,
#     cfg: AgglomerativeConfig = AgglomerativeConfig(),
# ) -> np.ndarray:
#     """
#     Agglomerative clustering using a cosine-distance threshold.
#     n_clusters is inferred from distance_threshold.
#     """
#     cl = AgglomerativeClustering(
#         n_clusters=None,
#         metric="cosine",
#         linkage=cfg.linkage,
#         distance_threshold=cfg.distance_threshold,
#     )
#     return cl.fit_predict(emb)


# def cluster_hdbscan(
#     emb: np.ndarray,
#     cfg: HDBSCANConfig = HDBSCANConfig(),
# ) -> np.ndarray:
#     """
#     HDBSCAN clustering. Returns -1 labels for outliers/noise.
#     Requires: pip install hdbscan
#     """
#     try:
#         import hdbscan  # type: ignore
#     except ImportError as e:
#         raise ImportError(
#             "hdbscan is not installed. Install with: pip install -U hdbscan"
#         ) from e

#     clusterer = hdbscan.HDBSCAN(
#         min_cluster_size=cfg.min_cluster_size,
#         min_samples=cfg.min_samples,
#         metric=cfg.metric,
#     )
#     return clusterer.fit_predict(emb)


# def cluster_texts(
#     texts: Union[pd.Series, List[str]],
#     method: ClusteringMethod = "hdbscan",
#     embed_cfg: EmbedConfig = EmbedConfig(),
#     kmeans_cfg: KMeansConfig = KMeansConfig(),
#     agglom_cfg: AgglomerativeConfig = AgglomerativeConfig(),
#     hdbscan_cfg: HDBSCANConfig = HDBSCANConfig(),
# ) -> Tuple[np.ndarray, np.ndarray]:
#     """
#     End-to-end: embeds texts then clusters.

#     Returns:
#         emb: embeddings (n, dim)
#         labels: cluster labels (n,)
#     """
#     emb = embed_texts(texts, embed_cfg)

#     if method == "kmeans":
#         labels = cluster_kmeans(emb, kmeans_cfg)
#     elif method == "agglomerative":
#         labels = cluster_agglomerative(emb, agglom_cfg)
#     elif method == "hdbscan":
#         labels = cluster_hdbscan(emb, hdbscan_cfg)
#     else:
#         raise ValueError(f"Unknown method: {method}")

#     return emb, labels


# # ---------------------------
# # Evaluation / Inspection helpers
# # ---------------------------

# def silhouette_cosine(emb: np.ndarray, labels: np.ndarray) -> Optional[float]:
#     """
#     Compute silhouette score using cosine distance.
#     Returns None if silhouette is undefined (e.g., 1 cluster, all noise).
#     """
#     unique = np.unique(labels)
#     # silhouette requires at least 2 clusters and no trivial single cluster case
#     # If HDBSCAN returns many -1 (noise), silhouette is still defined if >= 2 clusters exist.
#     # But if all labels are -1 or only one unique label, it is undefined.
#     if len(unique) < 2:
#         return None
#     # If everything is noise, also undefined.
#     if len(unique) == 1 and unique[0] == -1:
#         return None
#     return float(silhouette_score(emb, labels, metric="cosine"))


# def avg_within_cluster_similarity(
#     emb: np.ndarray,
#     labels: np.ndarray,
#     skip_noise: bool = True,
# ) -> List[Tuple[int, float, int]]:
#     """
#     For each cluster, compute average pairwise cosine similarity (excluding diagonal).
#     Returns list of (cluster_id, avg_similarity, cluster_size) sorted descending by similarity.
#     """
#     results: List[Tuple[int, float, int]] = []
#     for c in np.unique(labels):
#         if skip_noise and c == -1:
#             continue
#         idx = np.where(labels == c)[0]
#         n = len(idx)
#         if n < 2:
#             continue
#         m = cosine_similarity(emb[idx])
#         # average of off-diagonal entries
#         avg_sim = float((m.sum() - n) / (n * (n - 1)))
#         results.append((int(c), avg_sim, int(n)))
#     results.sort(key=lambda x: x[1], reverse=True)
#     return results


# def cluster_representatives(
#     df: pd.DataFrame,
#     emb: np.ndarray,
#     labels: np.ndarray,
#     text_col: str = "text",
#     top_n: int = 5,
#     skip_noise: bool = True,
# ) -> Dict[int, pd.DataFrame]:
#     """
#     Return representative rows per cluster: the items most similar to the cluster centroid.
#     Output dict: cluster_id -> DataFrame of top_n rows [text_col, cluster, similarity_to_centroid]
#     """
#     if text_col not in df.columns:
#         raise KeyError(f"text_col='{text_col}' not in DataFrame columns: {list(df.columns)}")

#     out: Dict[int, pd.DataFrame] = {}
#     for c in np.unique(labels):
#         if skip_noise and c == -1:
#             continue
#         idx = np.where(labels == c)[0]
#         if len(idx) == 0:
#             continue
#         centroid = emb[idx].mean(axis=0, keepdims=True)
#         sims = cosine_similarity(emb[idx], centroid).ravel()
#         order = np.argsort(-sims)[:top_n]
#         chosen = idx[order]
#         tmp = df.iloc[chosen].copy()
#         tmp["cluster"] = labels[chosen]
#         tmp["sim_to_centroid"] = sims[order]
#         out[int(c)] = tmp[[text_col, "cluster", "sim_to_centroid"]]
#     return out


# def attach_labels(
#     df: pd.DataFrame,
#     labels: np.ndarray,
#     col_name: str = "cluster",
# ) -> pd.DataFrame:
#     """
#     Return a copy of df with labels attached.
#     """
#     if len(df) != len(labels):
#         raise ValueError(f"df length {len(df)} != labels length {len(labels)}")
#     out = df.copy()
#     out[col_name] = labels
#     return out


# # ---------------------------
# # Example usage (how to call)
# # ---------------------------

# def _example_usage() -> None:
#     """
#     This function is only an example. You can delete it.
#     """
#     df = pd.DataFrame(
#         {"text": [
#             "Deep learning methods for medical imaging.",
#             "Convolutional networks improve image diagnosis.",
#             "The stock market fell after the announcement.",
#             "Equities reacted strongly to interest rate news.",
#             "A new vaccine showed strong immune response in trials."
#         ]}
#     )

#     # 1) HDBSCAN (good default if you do not know k)
#     emb, labels = cluster_texts(
#         df["text"],
#         method="hdbscan",
#         embed_cfg=EmbedConfig(model_name="all-mpnet-base-v2", normalize=True),
#         hdbscan_cfg=HDBSCANConfig(min_cluster_size=2, min_samples=1),
#     )
#     df_labeled = attach_labels(df, labels)
#     print(df_labeled)

#     # Inspect quality
#     sil = silhouette_cosine(emb, labels)
#     print("silhouette (cosine):", sil)

#     # Get representatives
#     reps = cluster_representatives(df_labeled, emb, labels, text_col="text", top_n=2)
#     for cid, rep_df in reps.items():
#         print("\nCluster", cid)
#         print(rep_df)

#     # 2) Agglomerative threshold example
#     emb2, labels2 = cluster_texts(
#         df["text"],
#         method="agglomerative",
#         agglom_cfg=AgglomerativeConfig(distance_threshold=0.35),
#     )
#     print("\nAgglomerative labels:", labels2)

#     # 3) KMeans example
#     emb3, labels3 = cluster_texts(
#         df["text"],
#         method="kmeans",
#         kmeans_cfg=KMeansConfig(n_clusters=3),
#     )
#     print("\nKMeans labels:", labels3)


# if __name__ == "__main__":
#     _example_usage()
