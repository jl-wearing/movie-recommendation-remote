"""
Reusable plotting helpers for the movie-recommendation project.

Every function saves a PNG into images/ so the results can be embedded in the
README. Static images (matplotlib + seaborn) are used because they render
directly on GitHub, unlike interactive HTML plots.
"""
import os

import matplotlib
matplotlib.use('Agg')  # non-interactive backend; we only save files
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sns.set_theme(style='whitegrid')

# Anchor to the project root so images always land in <root>/images.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGES_DIR = os.path.join(PROJECT_ROOT, 'images')


def _save(fig, filename):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    path = os.path.join(IMAGES_DIR, filename)
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {path}')
    return path


def plot_rating_distribution(data, filename='rating_distribution.png'):
    """Bar chart of how often each rating value (0-5) appears in the matrix.

    Rating 0 (unrated) is shown separately on a log scale because it dwarfs the
    actual ratings, which makes the sparsity of the data obvious.
    """
    values, counts = np.unique(data, return_counts=True)
    labels = ['0 (unrated)' if v == 0 else str(v) for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=sns.color_palette('viridis', len(values)))
    ax.set_yscale('log')
    ax.set_xlabel('Rating value')
    ax.set_ylabel('Count (log scale)')
    ax.set_title('MovieLens 1M rating distribution')
    ax.bar_label(bars, fmt='{:,.0f}', padding=3, fontsize=9)
    return _save(fig, filename)


def plot_class_balance(Y, filename='class_balance.png'):
    """Bar chart of the positive/negative label split for the target movie."""
    n_pos = int((Y == 1).sum())
    n_neg = int((Y == 0).sum())
    total = n_pos + n_neg

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(['Like (1)', 'Dislike (0)'], [n_pos, n_neg],
                  color=['#2a9d8f', '#e76f51'])
    ax.set_ylabel('Number of users')
    ax.set_title('Class balance for target movie (ID 2858)')
    for bar, n in zip(bars, [n_pos, n_neg]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f'{n:,}\n({n / total:.0%})', ha='center', va='bottom',
                fontsize=10)
    ax.set_ylim(0, max(n_pos, n_neg) * 1.15)
    return _save(fig, filename)


def plot_confusion_matrix(cm, filename='confusion_matrix.png',
                          labels=('Dislike (0)', 'Like (1)')):
    """Heatmap of a 2x2 confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', cbar=False,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={'fontsize': 13})
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('Confusion matrix (MultinomialNB)')
    return _save(fig, filename)


def plot_roc_curve(false_pos_rate, true_pos_rate, auc,
                   filename='roc_curve.png'):
    """ROC curve with the random-guess baseline and the AUC in the legend."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(false_pos_rate, true_pos_rate, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
            label='Random guess (AUC = 0.5)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    return _save(fig, filename)


def plot_feature_comparison(results, filename='feature_comparison.png'):
    """Grouped bar chart of accuracy and AUC across feature sets.

    @param results: {feature_set_name: {'accuracy': float, 'auc': float}}
    """
    names = list(results.keys())
    accuracy = [results[n]['accuracy'] for n in names]
    auc = [results[n]['auc'] for n in names]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, accuracy, width, label='Accuracy',
                color='#4c72b0')
    b2 = ax.bar(x + width / 2, auc, width, label='AUC', color='#dd8452')
    ax.bar_label(b1, fmt='{:.3f}', padding=2, fontsize=9)
    ax.bar_label(b2, fmt='{:.3f}', padding=2, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.0)
    ax.set_title('Effect of extra features on the recommender')
    ax.legend(loc='lower right')
    return _save(fig, filename)


def plot_model_auc_comparison(nb_results, gbt_results,
                              filename='model_comparison.png'):
    """Grouped bar chart of AUC for two models across the same feature sets.

    @param nb_results:  {feature_set_name: {'auc': float, ...}}
    @param gbt_results: {feature_set_name: {'auc': float, ...}}
    """
    names = list(nb_results.keys())
    nb_auc = [nb_results[n]['auc'] for n in names]
    gbt_auc = [gbt_results[n]['auc'] for n in names]

    x = np.arange(len(names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.bar(x - width / 2, nb_auc, width, label='MultinomialNB',
                color='#4c72b0')
    b2 = ax.bar(x + width / 2, gbt_auc, width, label='GradientBoosting',
                color='#55a868')
    ax.bar_label(b1, fmt='{:.3f}', padding=2, fontsize=9)
    ax.bar_label(b2, fmt='{:.3f}', padding=2, fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('AUC')
    ax.set_ylim(0, 1.0)
    ax.set_title('AUC by model and feature set')
    ax.legend(loc='lower right')
    return _save(fig, filename)


def plot_cv_auc_heatmap(auc_record, k, filename='cv_auc_heatmap.png'):
    """Heatmap of mean cross-validated AUC over (alpha, fit_prior) grid.

    @param auc_record: {alpha: {fit_prior: summed_auc}}
    @param k: number of folds (to average the summed AUCs)
    """
    alphas = sorted(auc_record.keys())
    fit_priors = [True, False]
    matrix = np.array([[auc_record[a][fp] / k for fp in fit_priors]
                       for a in alphas])

    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(matrix, annot=True, fmt='.5f', cmap='YlGnBu',
                xticklabels=[str(fp) for fp in fit_priors],
                yticklabels=alphas, ax=ax, cbar_kws={'label': 'Mean AUC'})
    ax.set_xlabel('fit_prior')
    ax.set_ylabel('alpha (smoothing factor)')
    ax.set_title(f'{k}-fold cross-validated AUC')
    return _save(fig, filename)
