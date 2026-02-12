import math

def compute_h_index(citations_per_paper):
    """
    Compute the h-index for a group of publications.
    """
    h_index = sum(c >= i + 1 for i, c in enumerate(citations_per_paper))
    # # Sort the publications by number of citations
    # counts_grouped_by_paper = counts_grouped_by_paper.sort_values(col_name, ascending=False)
    
    # # Compute the h-index
    # h_index = 0
    # for i, row in counts_grouped_by_paper.iterrows():
    #     if row[col_name] > h_index:
    #         h_index += 1
    #     else:
    #         break
    
    return h_index


def compute_i10_index(citations_per_paper):
    """
    Compute the i10-index for a group of publications.
    """
    i10_index = sum(c >= 10 for c in citations_per_paper)
    return i10_index

def compute_e_index(df_author, col_publication, col_citation_from):
    """
    Compute e-index https://e-index.net/about/
    e = -1/N sum_{i=1}^{N} (c_i) log(c_i / c_{total})
    """
    
    N = df_author[col_publication].nunique()
    c_total = df_author[col_citation_from].shape[0]

    e = -1/N * sum([len(citations) * math.log(len(citations) / c_total) for paper, citations in df_author.groupby(col_publication)])
    return e
