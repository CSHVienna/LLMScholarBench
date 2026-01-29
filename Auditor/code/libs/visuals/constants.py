
from matplotlib import pyplot as plt
from libs.visuals.grid import PanelSpec
from libs.constants import EXPERIMENT_TASKS

# Define your panel order and labels
PANEL_INFRASTRUCTURE = [
    PanelSpec("validity_pct", r"Validity $\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("refusal_pct", r"Refusal", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("duplicates", r"Duplicate $\downarrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=False),
    PanelSpec("consistency", r"Consistency", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("factuality_author", r"Factuality $\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("connectedness", r"Connectedness", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("similarity", r"Similarity", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("diversity_gender", r"Diversity", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("parity_gender", r"Parity $\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
]

# Color palettes per group (your scheme)
tab20 = plt.get_cmap("tab20")
tab20c = plt.get_cmap("tab20c")
GROUP_COLORS = {
    "model_access": [tab20(0), tab20(1)][::-1],
    "model_size": [tab20c(i) for i in range(8, 12)][::-1],
    "model_class": [tab20(2), tab20(3)][::-1],
}

tab10 = plt.get_cmap("tab10")
TASK_COLORS = {c: tab10(i) for i, c in enumerate(EXPERIMENT_TASKS)}

GROUPS_ORDER = ["model_access", "model_size", "model_class"]
# Label mappings
GROUP_LABEL_MAP = {"model_access": "Access", "model_size": "Size", "model_class": "Reasoning"}
MODEL_KIND_LABEL_MAP = {
    "open": "Open", "proprietary": "Proprietary",
    # "S": "Small", "M": "Medium", "L": "Large", "XL": "Extra Large",
    "non-reasoning": "Disabled", "reasoning": "Enabled",
}

MODEL_KIND_ORDER_WITHIN_GROUP = {
    "model_access": ["open", "proprietary"],
    "model_size": ["S", "M", "L", "XL"],
    "model_class": ["non-reasoning", "reasoning"],
}

# Highlight rules for bar figure
METRIC_HIGHLIGHT_RULES = {
    "validity_pct": "max",
    "duplicates": "min",
    "factuality_author": "max",
    "parity_gender": "max",
}



# xlim = (0, 1.1)
# xticks = [0, 0.5, 1.0]
# PANELS_INFRASTRUCTURE = [
#     PanelSpec("validity_pct", r"Validity $\uparrow$", xlim=xlim, xticks=xticks, draw_ci=True),
#     PanelSpec("refusal_pct", "Refusal", xlim=xlim, xticks=xticks, value_fmt="{:.2f}", draw_ci=True),
#     PanelSpec("duplicates", r"Duplicate $\downarrow$", xlim=xlim, xticks=xticks, draw_ci=True),
#     PanelSpec("consistency", "Consistency", xlim=xlim, xticks=xticks, draw_ci=True),
#     PanelSpec("factuality_author", r"Factuality $\uparrow$", xlim=xlim, xticks=xticks, draw_ci=True),
#     PanelSpec("diversity_gender", "Diversity", xlim=xlim, xticks=xticks, draw_ci=True),
#     PanelSpec("parity_gender", r"Parity $\uparrow$", xlim=xlim, xticks=xticks, draw_ci=True),
# ]


# ylim = (0, 1)
# PANELS_TEMPERATURE = [
#     PanelSpec("validity_pct", r"Validity $\uparrow$", draw_ci=True, ylim=ylim),
#     PanelSpec("refusal_pct", "Refusal", value_fmt="{:.2f}", draw_ci=True, ylim=ylim),
#     PanelSpec("duplicates", r"Duplicate $\downarrow$", draw_ci=True, ylim=ylim),
#     PanelSpec("consistency", "Consistency", draw_ci=True, ylim=ylim),
#     PanelSpec("factuality_author", r"Factuality $\uparrow$", draw_ci=True),
#     PanelSpec("diversity_gender", "Diversity", draw_ci=True, ylim=ylim),
#     PanelSpec("parity_gender", r"Parity $\uparrow$", draw_ci=True, ylim=ylim),
# ]



# # Colors as you defined
# tab20 = plt.get_cmap("tab20")
# tab20c = plt.get_cmap("tab20c")
# COLORS_MODEL_GROUP = {
#     "model_access": [tab20(0), tab20(1)][::-1],
#     "model_size": [tab20c(i) for i in range(8, 12)][::-1],
#     "model_class": [tab20(2), tab20(3)][::-1],
# }

# MODEL_GROUPS_ORDER = ("model_access", "model_size", "model_class")

# MODEL_ORDER_WITHIN_GROUP = {
#         "model_access": ["open", "proprietary"],
#         "model_size": ["S", "M", "L", "XL"],
#         "model_class": ["non-reasoning", "reasoning"],
#     }

# MODEL_GROUP_LABEL_MAP = {
#     "model_access": "Access",
#     "model_size": "Size",
#     "model_class": "Reasoning",
# }

# MODEL_LABEL_MAP = {
#     "open": "Open",
#     "proprietary": "Proprietary",
#     "S": "Small",
#     "M": "Medium",
#     "L": "Large",
#     "XL": "Extra Large",
#     "non-reasoning": "Disabled",
#     "reasoning": "Enabled",
# }

# METRIC_HIGHLIGHT_RULES = {
#     "validity_pct": "max",
#     "duplicate_ratio": "min",
#     "factuality_author": "max",
#     "parity_gender": "max",
# }
