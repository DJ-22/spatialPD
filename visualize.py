import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
import seaborn as sns
from config import STRATEGY_MAPPING, STRATEGY_COLORS, NUM, ALLC, ALLD, TFT, PAVLOV, GRUDGER, RANDOM
from experiment import extract_time_series


STRATEGY_ORDER = [ALLC, TFT, PAVLOV, GRUDGER, RANDOM, ALLD]

def _apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3
    })

def plot_time_series(
    log,
    title=None,
    mu=None,
    lam=None,
    b=None,
    ax=None,
    save_path=None
):
    _apply_style()
    series = extract_time_series(log, field="fractions")
    gen = np.arange(len(log))
    
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 5))
        
    for s in STRATEGY_ORDER:
        ax.plot(gen, series[s], label=STRATEGY_MAPPING[s], color=STRATEGY_COLORS[s], linewidth=1.8)
    
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Fraction of Population", fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, gen[-1])
    ax.legend(loc="upper right", fontsize=9, framealpha=0.7)
    
    if title is None:
        parts = []
        if mu is not None:
            parts.append(f"μ={mu:.3f}")
        if lam is not None:
            parts.append(f"λ={lam:.3f}")
        if b is not None:
            parts.append(f"b={b:.1f}")
        
        title = "Strategy Composition over Time"
        if parts:
            title += " (" + ", ".join(parts) + ")"
    ax.set_title(title, fontsize=13, pad=10)
        
    if own_fig:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig, ax
    
    return ax


def plot_heatmap(
    heatmap_result,
    title=None,
    save_path=None,
    ax=None,
    vmin=0.0,
    vmax=1.0,
    annot=True
):
    _apply_style()
    mu_values = heatmap_result["mu_values"]
    lambda_values = heatmap_result["lambda_values"]
    data = heatmap_result["heatmap"]
    b = heatmap_result.get("b")

    x_labels = [str(m) for m in mu_values]
    y_labels = [str(l) for l in lambda_values]
 
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))
 
    sns.heatmap(
        data,
        ax=ax,
        xticklabels=x_labels,
        yticklabels=y_labels,
        vmin=vmin, vmax=vmax,
        cmap="RdYlGn",
        annot=annot,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Equilibrium Cooperation Fraction"}
    )
 
    ax.set_xlabel("Mutation Rate (μ)", fontsize=12)
    ax.set_ylabel("Aging Rate (λ)",    fontsize=12)
 
    if title is None:
        title = "Equilibrium Cooperation"
        if b is not None:
            title += f"  (b={b})"
    ax.set_title(title, fontsize=13, pad=10)
 
    if own_fig:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig, ax
    
    return ax


def plot_snapshots(
    snapshots,
    title_prefix="Generation",
    save_path=None
):
    _apply_style()
    gens = sorted(snapshots.keys())
    n = len(gens)

    cmap_colors = ["#ffffff"] + [STRATEGY_COLORS[s] for s in range(NUM)]
    cmap  = ListedColormap(cmap_colors)
    bounds = [-1.5] + [s + 0.5 for s in range(-1, NUM)]
    norm  = BoundaryNorm(bounds, cmap.N)
 
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]
 
    for ax, gen in zip(axes, gens):
        grid = snapshots[gen]
        ax.imshow(grid, cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{title_prefix} {gen}", fontsize=11)
        ax.axis("off")

    legend_patches = [
        mpatches.Patch(color=STRATEGY_COLORS[s], label=STRATEGY_MAPPING[s])
        for s in range(NUM)
    ]
    legend_patches.append(mpatches.Patch(color="white", label="Empty", ec="grey"))
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=NUM + 1,
        fontsize=9,
        framealpha=0.8,
        bbox_to_anchor=(0.5, -0.05),
    )
 
    fig.suptitle("Spatial Distribution of Strategies", fontsize=13, y=1.02)
    plt.tight_layout()
 
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig, axes


def plot_multi_timeseries(
    results,
    labels=None,
    ncols=2,
    save_path=None
):
    _apply_style()
    n = len(results)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
 
    for idx, log in enumerate(results):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        title = labels[idx] if labels else None
        plot_time_series(log, title=title, ax=ax)

    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)
 
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    
    return fig, axes


def plot_coop_rate(log, mu=None, lam=None, b=None, ax=None, save_path=None):
    _apply_style()
    generations = np.arange(len(log))
    coop = np.array([entry["coop_rate"] for entry in log])
 
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(9, 4))
 
    ax.plot(generations, coop, color="#27ae60", linewidth=1.8)
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Cooperation Rate", fontsize=12)
    ax.set_ylim(0, 1)
 
    parts = []
    if mu  is not None: 
        parts.append(f"μ={mu}")
    if lam is not None: 
        parts.append(f"λ={lam}")
    if b   is not None: 
        parts.append(f"b={b}")
    
    title = "Cooperation Rate over Time"
    if parts:
        title += "  (" + ", ".join(parts) + ")"
    ax.set_title(title, fontsize=13)
 
    if own_fig:
        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        
        return fig, ax
    
    return ax
