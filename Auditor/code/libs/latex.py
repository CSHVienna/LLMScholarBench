import re
import numpy as np
import pandas as pd


group_map_infrastructure = {
    "access": ["open", "proprietary"],
    "size": ["S", "M", "L", "XL"],
    "reasoning": ["non-reasoning", "reasoning"],
}

metric_higher_better = {
    "validity_pct", "factuality_author", "parity_gender"
}

metric_lower_better = {"duplicates"} 


num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

def extract_mean(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float, np.number)):
        return float(x)
    m = num_re.search(str(x))
    return float(m.group(0)) if m else np.nan

def bold_best_within_groups(df, group_map, higher_better, lower_better):
    out = df.copy()

    # numeric view used only for comparisons
    num = out.applymap(extract_mean)

    for _, rows in group_map.items():
        sub = num.loc[rows]

        for col in out.columns:
            if col in higher_better:
                best = sub[col].max()
                mask = (sub[col] == best) & sub[col].notna()
            elif col in lower_better:
                best = sub[col].min()
                mask = (sub[col] == best) & sub[col].notna()
            else:
                continue

            # apply bold to the original (string) table
            for r in sub.index[mask]:
                out.loc[r, col] = r"\textbf{" + str(out.loc[r, col]) + "}"

    return out