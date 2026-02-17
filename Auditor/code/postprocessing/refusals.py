from libs import helpers
from libs.text import helpers as text
from sentence_transformers import util
from sentence_transformers import SentenceTransformer

from libs import io
from libs import constants

def get_refusals(df_summaries: io.pd.DataFrame):
    if 'is_refusal' not in df_summaries.columns:
        # df_summaries.loc[:, 'is_refusal'] = df_summaries.apply(lambda row: helpers.detect_refusal(row), axis=1)
        raise ValueError("is_refusal column not found in summaries")
    df_refusals = df_summaries.query("is_refusal == 1 and (@io.pd.notna(result_original_message) or @io.pd.notna(result_original_output))").copy()
    df_refusals.loc[:, 'refusal_text'] = df_refusals.apply(lambda row: helpers.get_refusal_text(row), axis=1)
    df_refusals.loc[:, 'norm_text'] = df_refusals.refusal_text.map(text.normalize_text)
    return df_refusals

def get_unique_refusals(df_refusals: io.pd.DataFrame):
    df_refusals_unique_texts = df_refusals[['norm_text']].copy()
    df_refusals_unique_texts.norm_text = df_refusals_unique_texts.norm_text.str.lower()
    df_refusals_unique_texts.drop_duplicates(subset=['norm_text'], inplace=True)
    df_refusals_unique_texts.reset_index(drop=True, inplace=True)
    return df_refusals_unique_texts

def get_corpus_embeddings(embedder: SentenceTransformer, corpus: list):
    corpus_embeddings = embedder.encode_document(corpus, convert_to_tensor=True)
    return corpus_embeddings

def do_clustering(embedder: SentenceTransformer, corpus_embeddings: io.np.ndarray, df_refusals_unique_texts: io.pd.DataFrame):
    top_k = df_refusals_unique_texts.shape[0]
    mapping_cluster_names = {}
    for gid, (group, queries) in enumerate(constants.REFUSAL_CLUSTER_SEEDS.items()):
        cg = f"g{gid}"
        mapping_cluster_names[cg] = group
        nqueries = len(queries)
        for qid, query in enumerate(queries):
            cq = f"q{qid}"
            cid = f"{cg}_{cq}"

            query_embedding = embedder.encode_query(query.lower(), convert_to_tensor=True)

            hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)
            hits = io.pd.DataFrame(hits[0])
            hits.rename(columns={'score':cid}, inplace=True)

            df_refusals_unique_texts = df_refusals_unique_texts.join(hits.set_index('corpus_id'))

        cols = [c for c in df_refusals_unique_texts.columns if c.startswith(cg)]
        df_refusals_unique_texts.loc[:, cg] = df_refusals_unique_texts[cols].sum(axis=1) / nqueries

    # final clustering
    df_refusals_unique_texts.loc[:, 'cluster_sim'] = df_refusals_unique_texts[mapping_cluster_names.keys()].max(axis=1)
    df_refusals_unique_texts.loc[:, 'cluster_id'] = df_refusals_unique_texts[mapping_cluster_names.keys()].idxmax(axis=1)
    df_refusals_unique_texts.loc[:, 'cluster_name'] = df_refusals_unique_texts.cluster_id.map(mapping_cluster_names)

    return df_refusals_unique_texts, mapping_cluster_names