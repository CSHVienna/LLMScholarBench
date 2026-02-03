
from matplotlib import pyplot as plt
from libs.visuals.grid import PanelSpec
from libs.constants import EXPERIMENT_TASKS, EXPERIMENT_TASKS_2TWINS
from libs.constants import TASK_PARAMS_BY_TASK

# Define your panel order and labels
PANELS_METRICS_GROUPS = [
    PanelSpec("refusal_pct", r"Refusal", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("validity_pct", r"Validity $\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("duplicates", r"Duplicate $\downarrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("consistency", r"Consistency", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("factuality_author", r"Factuality $\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("connectedness", r"Connectedness", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("similarity_pca", r"Similarity", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("diversity_gender", r"Diversity", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
    PanelSpec("parity_gender", r"Parity $\uparrow$", xlim=(0, 1), xticks=[0, 0.5, 1.0], draw_ci=True),
]

ylim_t = (0, 1.05)
xlim_t = (0, 2.0)
xticks_t = [0, 0.5, 1.0, 1.5, 2.0]
PANELS_METRICS = [
    PanelSpec("refusal_pct", r"Refusal", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("validity_pct", r"Validity $\uparrow$", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("duplicates", r"Duplicate $\downarrow$", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("consistency", r"Consistency", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("factuality_author", r"Factuality $\uparrow$", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("connectedness", r"Connectedness", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("similarity_pca", r"Similarity", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
    PanelSpec("diversity_gender", r"Diversity", ylim=(0, 1), xlim=(0, 2.0), xticks=[0, 0.5, 1.0, 1.5, 2.0], draw_ci=True),
    PanelSpec("parity_gender", r"Parity $\uparrow$", ylim=ylim_t, xlim=xlim_t, xticks=xticks_t, draw_ci=True),
]


ylim=(-0.1, 1.1)
yticks=[0, .25, .5, .75, 1.0]
PANELS_METRICS_BEFORE_AFTER = [
    PanelSpec("refusal_pct", r"Refusal", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("validity_pct", r"Validity $\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("duplicates", r"Duplicate $\downarrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("consistency", r"Consistency", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("factuality_author", r"Factuality $\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("connectedness", r"Connectedness", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("similarity_pca", r"Similarity", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("diversity_gender", r"Diversity", ylim=ylim, yticks=yticks, draw_ci=True),
    PanelSpec("parity_gender", r"Parity $\uparrow$", ylim=ylim, yticks=yticks, draw_ci=True),
]


# Color palettes per group (your scheme)
tab20 = plt.get_cmap("tab20")
tab20c = plt.get_cmap("tab20c")
GROUP_COLORS = {
    "model_access": [tab20(0), tab20(1)][::-1],
    "model_size": [tab20c(i) for i in range(8, 12)][::-1],
    "model_class": [tab20(2), tab20(3)][::-1],
}

BIASED_PROMPT_GENDER_COLORS = {
    "top_100_bias_gender_equal": "tab:green",
    "top_100_bias_gender_female": "tab:red",
    "top_100_bias_gender_male": "tab:blue",
    "top_100_bias_gender_neutral": "tab:grey",
}

tab10 = plt.get_cmap("tab10")
TASK_COLORS = {c: tab10(i) for i, c in enumerate(EXPERIMENT_TASKS)}
TASK_COLORS_2TWINS = {c: tab10(i + (1 if i==len(EXPERIMENT_TASKS_2TWINS)-1 else 0)) for i, c in enumerate(EXPERIMENT_TASKS_2TWINS)}

tab20 = plt.get_cmap("tab20")
TASK_PARAM_COLORS = {(task_name, task_param):tab20((i*2) + j) for i, (task_name, task_params) in enumerate(TASK_PARAMS_BY_TASK.items()) for j, task_param in enumerate(task_params)}

