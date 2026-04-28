import argparse
import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from config import MUTATION_RATE, LAMBDA, B, NUM_GEN, NUM_SEEDS, BASE_SEED, GRID_WIDTH, GRID_HEIGHT, MUTATION_VALUES, LAMBDA_VALUES, B_VALUES
from experiment import single_run, averaged_run, mu_lambda_sweep, full_sweep, extract_time_series, CHECKPOINT_DIR
from visualize import plot_time_series, plot_heatmap, plot_snapshots, plot_multi_timeseries, plot_coop_rate


OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
_RUN_DIR = None

def out(filename):
    return os.path.join(_RUN_DIR, filename)


def make_prefix(b, mu, lam, seed):
    b_str = f"b{b:.1f}".replace(".", "-")
    mu_str = f"mu{mu:.3f}".replace(".", "-")
    lam_str = f"lam{lam:.3f}".replace(".", "-")
    return f"{b_str}_{mu_str}_{lam_str}_seed{seed}"


def save_run_plots(grid, b, mu, lam, seed):
    prefix = make_prefix(b, mu, lam, seed)

    fig, _ = plot_time_series(
        grid.log, mu=mu, lam=lam, b=b,
        save_path=out(f"{prefix}_timeseries.png"),
    )
    plt.close(fig)

    fig, _ = plot_coop_rate(
        grid.log, mu=mu, lam=lam, b=b,
        save_path=out(f"{prefix}_coop_rate.png"),
    )
    plt.close(fig)

    if hasattr(grid, "snapshots") and grid.snapshots:
        fig, _ = plot_snapshots(
            grid.snapshots,
            save_path=out(f"{prefix}_snapshots.png"),
        )
        plt.close(fig)

    print(f"  Saved: {prefix}_{{timeseries,coop_rate,snapshots}}.png")


def mode_single(args):
    print(f"Single run μ={args.mu:.3f} λ={args.lam:.3f} b={args.b:.1f} "
          f"G={args.generations} seed={args.seed}")

    snap_gens = [0, 500, 1000, args.generations]
    checkpoint_path = os.path.join(CHECKPOINT_DIR,
                                   f"{make_prefix(args.b, args.mu, args.lam, args.seed)}.pkl")

    grid = single_run(
        mu=args.mu, lam=args.lam, b=args.b,
        num_generations=args.generations,
        seed=args.seed,
        width=args.width, height=args.height,
        progress=True,
        snapshot_gens=snap_gens,
        checkpoint_path=checkpoint_path,
    )

    save_run_plots(grid, args.b, args.mu, args.lam, args.seed)
    print(f"Equilibrium cooperation — strategy-based: {grid.equilibrium_coop():.3f}, "
          f"action-based: {grid.equilibrium_action_coop():.3f}")


def mode_averaged(args):
    print(f"Averaged run  μ={args.mu}  λ={args.lam}  b={args.b}  "
          f"seeds={args.num_seeds}")

    result = averaged_run(
        mu=args.mu, lam=args.lam, b=args.b,
        num_generations=args.generations,
        num_seeds=args.num_seeds,
        base_seed=args.seed,
        width=args.width, height=args.height,
        progress=True,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    prefix = make_prefix(args.b, args.mu, args.lam, "avg")

    fig, _ = plot_time_series(
        result["log"], mu=args.mu, lam=args.lam, b=args.b,
        save_path=out(f"{prefix}_timeseries.png"),
    )
    plt.close(fig)

    fig, _ = plot_coop_rate(
        result["log"], mu=args.mu, lam=args.lam, b=args.b,
        save_path=out(f"{prefix}_coop_rate.png"),
    )
    plt.close(fig)

    print(f"Saved: {prefix}_{{timeseries,coop_rate}}.png")
    print(f"Equilibrium cooperation — strategy-based: {result['eq_coop']:.3f}, "
          f"action-based: {result['eq_action_coop']:.3f}")


def mode_sweep(args):
    print(f"μ×λ sweep  b={args.b}  seeds={args.num_seeds}  G={args.generations}")

    snap_gens = [0, args.generations // 2, args.generations]
    combos = [(lam, mu) for lam in LAMBDA_VALUES for mu in MUTATION_VALUES]
    total = len(combos)
    heatmap_data = np.zeros((len(LAMBDA_VALUES), len(MUTATION_VALUES)))
    heatmap_action = np.zeros((len(LAMBDA_VALUES), len(MUTATION_VALUES)))

    for idx, (lam, mu) in enumerate(combos):
        i = LAMBDA_VALUES.index(lam)
        j = MUTATION_VALUES.index(mu)
        print(f"[{idx+1}/{total}] μ={mu} λ={lam} b={args.b}", flush=True)

        eq_coops = []
        eq_action_coops = []
        for s in range(args.num_seeds):
            seed = args.seed + s * 4
            checkpoint_path = os.path.join(CHECKPOINT_DIR,
                                           f"{make_prefix(args.b, mu, lam, seed)}.pkl")
            grid = single_run(
                mu=mu, lam=lam, b=args.b,
                num_generations=args.generations,
                seed=seed,
                width=args.width, height=args.height,
                progress=True,
                snapshot_gens=snap_gens,
                checkpoint_path=checkpoint_path,
            )
            if args.save_per_seed_plots:
                save_run_plots(grid, args.b, mu, lam, seed)
            eq_coops.append(grid.equilibrium_coop())
            eq_action_coops.append(grid.equilibrium_action_coop())

        heatmap_data[i, j] = float(np.mean(eq_coops))
        heatmap_action[i, j] = float(np.mean(eq_action_coops))

    result = {
        "mu_values":     MUTATION_VALUES,
        "lambda_values": LAMBDA_VALUES,
        "heatmap":       heatmap_data,
        "b":             args.b,
    }
    result_action = {
        "mu_values":     MUTATION_VALUES,
        "lambda_values": LAMBDA_VALUES,
        "heatmap":       heatmap_action,
        "b":             args.b,
    }

    fig, _ = plot_heatmap(result, save_path=out(f"heatmap_b{args.b:.1f}_strategy.png"))
    plt.close(fig)
    fig, _ = plot_heatmap(
        result_action,
        title=f"Equilibrium Cooperative Actions  (b={args.b})",
        save_path=out(f"heatmap_b{args.b:.1f}_action.png"),
    )
    plt.close(fig)
    print(f"Saved: {out(f'heatmap_b{args.b:.1f}_{{strategy,action}}.png')}")
    print("strategy-based:")
    print(np.array2string(heatmap_data, precision=3))
    print("action-based:")
    print(np.array2string(heatmap_action, precision=3))


def mode_full_sweep(args):
    print(f"Full sweep over b={B_VALUES}  seeds={args.num_seeds}  G={args.generations}")

    snap_gens = [0, args.generations // 2, args.generations]
    all_heatmap_results = []
    all_heatmap_action = []

    for b in B_VALUES:
        print(f"\n=== b = {b:.1f} ===", flush=True)
        heatmap_data = np.zeros((len(LAMBDA_VALUES), len(MUTATION_VALUES)))
        heatmap_action = np.zeros((len(LAMBDA_VALUES), len(MUTATION_VALUES)))
        combos = [(lam, mu) for lam in LAMBDA_VALUES for mu in MUTATION_VALUES]
        total = len(combos)

        for idx, (lam, mu) in enumerate(combos):
            i = LAMBDA_VALUES.index(lam)
            j = MUTATION_VALUES.index(mu)
            print(f"  [{idx+1}/{total}] μ={mu} λ={lam} b={b}", flush=True)

            eq_coops = []
            eq_action_coops = []
            for s in range(args.num_seeds):
                seed = args.seed + s * 4
                checkpoint_path = os.path.join(CHECKPOINT_DIR,
                                               f"{make_prefix(b, mu, lam, seed)}.pkl")
                grid = single_run(
                    mu=mu, lam=lam, b=b,
                    num_generations=args.generations,
                    seed=seed,
                    width=args.width, height=args.height,
                    progress=True,
                    snapshot_gens=snap_gens,
                    checkpoint_path=checkpoint_path,
                )
                if args.save_per_seed_plots:
                    save_run_plots(grid, b, mu, lam, seed)
                eq_coops.append(grid.equilibrium_coop())
                eq_action_coops.append(grid.equilibrium_action_coop())

            heatmap_data[i, j] = float(np.mean(eq_coops))
            heatmap_action[i, j] = float(np.mean(eq_action_coops))
            print(f"    eq_coop = {np.mean(eq_coops):.3f} ± {np.std(eq_coops):.3f}  "
                  f"(per-seed: {[f'{x:.2f}' for x in eq_coops]})", flush=True)

        res = {
            "mu_values":     MUTATION_VALUES,
            "lambda_values": LAMBDA_VALUES,
            "heatmap":       heatmap_data,
            "b":             b,
        }
        res_action = {
            "mu_values":     MUTATION_VALUES,
            "lambda_values": LAMBDA_VALUES,
            "heatmap":       heatmap_action,
            "b":             b,
        }
        all_heatmap_results.append(res)
        all_heatmap_action.append(res_action)

        fig, _ = plot_heatmap(res, save_path=out(f"heatmap_b{b:.1f}_strategy.png"))
        plt.close(fig)
        fig, _ = plot_heatmap(
            res_action,
            title=f"Equilibrium Cooperative Actions  (b={b})",
            save_path=out(f"heatmap_b{b:.1f}_action.png"),
        )
        plt.close(fig)
        print(f"  Saved: {out(f'heatmap_b{b:.1f}_{{strategy,action}}.png')}")


    def _combined(results, suffix, title_word):
        n  = len(results)
        nc = min(3, n)
        nr = (n + nc - 1) // nc
        fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 5 * nr))
        axes_flat = axes.flatten() if n > 1 else [axes]

        for ax, res in zip(axes_flat, results):
            plot_heatmap(
                res, ax=ax, annot=True,
                title=f"{title_word}  (b={res['b']})",
            )

        for ax in axes_flat[n:]:
            ax.set_visible(False)

        plt.tight_layout()
        fig.savefig(out(f"heatmaps_all_b_{suffix}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out(f'heatmaps_all_b_{suffix}.png')}")

    _combined(all_heatmap_results, "strategy", "Equilibrium Cooperation")
    _combined(all_heatmap_action,  "action",   "Equilibrium Cooperative Actions")


def mode_compare(args):
    combos = [
        (0.001, 0.01, "Low μ, Low λ"),
        (0.05,  0.2,  "High μ, High λ"),
        (0.01,  0.05, "Intermediate μ, λ (default)"),
        (0.01,  0.05, f"Intermediate — high b={1.9}"),
    ]
    b_overrides = [args.b, args.b, args.b, 1.9]

    logs, labels = [], []
    for (mu, lam, label), b_val in zip(combos, b_overrides):
        print(f"  Running: {label}")
        res = averaged_run(
            mu=mu, lam=lam, b=b_val,
            num_generations=args.generations,
            num_seeds=args.num_seeds,
            base_seed=args.seed,
            width=args.width, height=args.height,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        logs.append(res["log"])
        labels.append(label)

    fig, _ = plot_multi_timeseries(
        logs, labels=labels, ncols=2,
        save_path=out("comparison.png"),
    )
    plt.close(fig)
    print(f"Saved: {out('comparison.png')}")


def build_parser():
    p = argparse.ArgumentParser(
        description="Spatial Prisoner's Dilemma — Evolution of Cooperation"
    )
    p.add_argument("--mode", choices=["single", "averaged", "sweep", "full_sweep", "compare"],
                   default="single", help="Run mode (default: single)")
    p.add_argument("--mu",          type=float, default=MUTATION_RATE, help=f"Mutation rate (default {MUTATION_RATE})")
    p.add_argument("--lam",         type=float, default=LAMBDA,        help=f"Aging rate (default {LAMBDA})")
    p.add_argument("--b",           type=float, default=B,             help=f"Temptation payoff (default {B})")
    p.add_argument("--generations", type=int,   default=NUM_GEN,       help=f"Generations (default {NUM_GEN})")
    p.add_argument("--num_seeds",   type=int,   default=NUM_SEEDS,     help=f"Seeds per combo (default {NUM_SEEDS})")
    p.add_argument("--seed",        type=int,   default=BASE_SEED,     help=f"Base random seed (default {BASE_SEED})")
    p.add_argument("--width",       type=int,   default=GRID_WIDTH,    help=f"Grid width (default {GRID_WIDTH})")
    p.add_argument("--height",      type=int,   default=GRID_HEIGHT,   help=f"Grid height (default {GRID_HEIGHT})")
    p.add_argument("--save_per_seed_plots", action="store_true",
                   help="In sweep modes, also save per-seed timeseries/coop_rate/snapshot PNGs (default off)")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Subdirectory under output/ to save plots in. "
                        "Defaults to a timestamped folder per invocation so prior runs aren't overwritten.")

    return p


def main():
    global _RUN_DIR
    args = build_parser().parse_args()

    if args.output_dir:
        _RUN_DIR = os.path.join(OUTPUT_DIR, args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _RUN_DIR = os.path.join(OUTPUT_DIR, f"run_{ts}_{args.mode}")
    os.makedirs(_RUN_DIR, exist_ok=True)
    print(f"Saving outputs to: {_RUN_DIR}")

    dispatch = {
        "single":     mode_single,
        "averaged":   mode_averaged,
        "sweep":      mode_sweep,
        "full_sweep": mode_full_sweep,
        "compare":    mode_compare,
    }
    dispatch[args.mode](args)
    print("\nDone.")


if __name__ == "__main__":
    main()
