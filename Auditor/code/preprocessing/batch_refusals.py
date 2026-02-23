# export PYTHONPATH="${PYTHONPATH}:."

import argparse

from libs import io
from libs import constants
from postprocessing import refusals
from sentence_transformers import SentenceTransformer


def _get_summaries(summaries_dirs: str, summaries_sources: str):
    df_summaries = io.pd.DataFrame()
    for summaries_dir, summaries_source in zip(summaries_dirs, summaries_sources):
        print(f"Loading summaries from {summaries_dir} for {summaries_source}")
        tmp = io.pd.concat([io.read_csv(io.path_join(summaries_dir, f"experiments_{model}.csv"), low_memory=False) for model in constants.LLMS], ignore_index=True)
        tmp.loc[:, 'source'] = summaries_source
        df_summaries = io.pd.concat([df_summaries, tmp], ignore_index=True)
    return df_summaries


def run(summaries_dirs: str, summaries_sources: str, output_dir: str):
    
    results_dir = io.path_join(output_dir, 'refusals')
    io.validate_path(results_dir)

    # required file names
    fn_corpus = io.path_join(results_dir, constants.FN_REFUSALS_CORPUS)
    fn_corpus_embeddings = io.path_join(results_dir, constants.FN_REFUSALS_CORPUS_EMBEDDINGS)
    fn_refusals_clustered = io.path_join(results_dir, constants.FN_REFUSALS_CLUSTERED)
    fn_cluster_names_mapping = io.path_join(results_dir, constants.FN_REFUSALS_CLUSTER_NAMES_MAPPING)

    # load summaries
    df_summaries = _get_summaries(summaries_dirs, summaries_sources)

    # normalize refusal text
    df_refusals = refusals.get_refusals(df_summaries)
    io.printf(f"Number of refusals: {df_refusals.shape[0]} ({df_summaries.is_refusal.value_counts()})")
    
    # creating corpus (unique refusal texts)
    df_refusals_unique_texts = refusals.get_unique_refusals(df_refusals)
    corpus = df_refusals_unique_texts.norm_text.tolist()
    io.save_dicts_to_text_file(corpus, fn_corpus)

    # Model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    corpus_embeddings = refusals.get_corpus_embeddings(embedder, corpus)
    io.save_numpy_array(corpus_embeddings, fn_corpus_embeddings)
    io.printf(f"Array successfully saved to {fn_corpus_embeddings}")

    # Inference
    df_refusals_unique_texts, mapping_cluster_names = refusals.do_clustering(embedder, corpus_embeddings, df_refusals_unique_texts)

    # Adding others cluster
    key = f'g{len(mapping_cluster_names) + 1}'
    mapping_cluster_names |= {key: constants.REFUSAL_CLUSTER_OTHER}

    q = [f"norm_text.str.contains('{k}')" for k in constants.REFUSAL_KEYWORDS_NOT_IN_OTHER]
    q = "norm_text.str.len() < @constants.REFUSAL_CLUSTER_OTHER_MAX_LENGTH and ~(" + "|".join(q) + ")"

    others = df_refusals_unique_texts.query(q).index
    df_refusals_unique_texts.loc[others, 'cluster_id'] = key
    df_refusals_unique_texts.loc[others, 'cluster_name'] = mapping_cluster_names[key]
    io.printf(f"Number of refusals per cluster: {df_refusals_unique_texts.cluster_id.value_counts()}")
    
    io.save_json_file(mapping_cluster_names, fn_cluster_names_mapping)
    io.printf(f"Saving cluster names mapping to {fn_cluster_names_mapping}")

    # assigning cluster to all refusals
    df_refusals_clustered_all = df_refusals.merge(df_refusals_unique_texts, on='norm_text', how='left') # all instances
    df_refusals_clustered_all.to_csv(fn_refusals_clustered)
    io.printf(f"Saving refusals with cluster information to {fn_refusals_clustered} ({df_refusals_clustered_all.shape[0]} refusals)")

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries_dirs", required=True, type=str, nargs="+", help="Directories where the summaries are stored")
    parser.add_argument("--summaries_sources", required=True, type=str, nargs="+", help="Prefix of the summaries sources (e.g., temperature, interventions)")
    parser.add_argument("--output_dir", required=True, type=str, help="Directory where the output files will be saved")
    args = parser.parse_args()

    print('=' * 10)
    print(f"Running {__file__} with:")
    for k, v in args.__dict__.items():
        print(f"{k}: {v}")
    print('=' * 10)

    run(args.summaries_dirs, args.summaries_sources, args.output_dir)

    