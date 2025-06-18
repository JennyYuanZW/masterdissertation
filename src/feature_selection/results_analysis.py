import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import roc_curve, precision_recall_curve, auc


def plot_roc_curve(runs, ax):
    default_blue = mcolors.CSS4_COLORS["steelblue"]
    light_blue = mcolors.to_rgba("steelblue", alpha=0.3)

    tprs = []
    fpr_grid = np.linspace(0, 1, 200)

    for run in runs:
        fpr, tpr, _ = roc_curve(run["y_true"], run["y_prob"])
        tpr_interp = np.interp(fpr_grid, fpr, tpr)
        tprs.append(tpr_interp)

    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    mean_auc = auc(fpr_grid, mean_tpr)

    ax.plot(
        fpr_grid,
        mean_tpr,
        color=default_blue,
        linewidth=2.0,
        label=f"Mean ROC (AUC = {mean_auc:.3f})",
    )

    ax.fill_between(
        fpr_grid,
        mean_tpr - std_tpr,
        mean_tpr + std_tpr,
        color=light_blue,
        label="±1 SD",
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.7)

    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curve Across Runs", fontsize=16)
    ax.legend(loc="lower right", fontsize=12)
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")

    return mean_auc


def plot_pr_curve(runs, ax):

    default_orange = mcolors.CSS4_COLORS["darkorange"]
    light_orange = mcolors.to_rgba("darkorange", alpha=0.3)

    precisions = []
    recall_grid = np.linspace(0, 1, 200)

    for run in runs:
        prec, rec, _ = precision_recall_curve(run["y_true"], run["y_prob"])
        prec_interp = np.interp(recall_grid, rec[::-1], prec[::-1])
        precisions.append(prec_interp)

    mean_prec = np.mean(precisions, axis=0)
    std_prec = np.std(precisions, axis=0)
    mean_auprc = auc(recall_grid, mean_prec)

    ax.plot(
        recall_grid,
        mean_prec,
        color=default_orange,
        linewidth=2.0,
        label=f"Mean PR (AUPRC = {mean_auprc:.3f})",
    )

    ax.fill_between(
        recall_grid,
        mean_prec - std_prec,
        mean_prec + std_prec,
        color=light_orange,
        label="±1 SD",
    )

    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title("Precision-Recall Curve Across Runs", fontsize=16)
    ax.legend(loc="lower left", fontsize=12)
    ax.grid(alpha=0.2)
    ax.set_aspect("equal", adjustable="box")

    return mean_auprc


if __name__ == "__main__":

    pickle_path = "results/baseline_results.pkl"

    with open(pickle_path, "rb") as f:
        results = pickle.load(f)

    prediction_runs = results["predictions"]

    fig_roc, ax_roc = plt.subplots(figsize=(7, 7))
    plot_roc_curve(prediction_runs, ax_roc)
    fig_roc.tight_layout()
    plt.show()

    fig_pr, ax_pr = plt.subplots(figsize=(7, 7))
    plot_pr_curve(prediction_runs, ax_pr)
    fig_pr.tight_layout()
    plt.show()

    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc", "auprc"]

    for m in metrics:
        values = np.array(results.get(m, []))
        if values.size > 0:
            mean_val = f"{np.mean(values):.4f}"
            std_val = f"{np.std(values):.4f}"
            print(f"{m:<12} | {mean_val:<10} | {std_val:<10}")
