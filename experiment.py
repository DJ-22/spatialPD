import os
import numpy as np
from tqdm import tqdm
from simulation import Grid
from config import MUTATION_VALUES, LAMBDA_VALUES, B_VALUES, NUM_SEEDS, BASE_SEED, NUM_GEN, GRID_WIDTH, GRID_HEIGHT, B, MUTATION_RATE, LAMBDA, NUM

CHECKPOINT_DIR = "checkpoints"


def single_run(
    mu=MUTATION_RATE,
    lam=LAMBDA,
    b=B,
    num_generations=NUM_GEN,
    seed=BASE_SEED,
    width=GRID_WIDTH,
    height=GRID_HEIGHT,
    progress=True,
    snapshot_gens=None,
    checkpoint_path=None,
    checkpoint_every=100,
):
    snap_set = set(snapshot_gens or [])

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Resuming from checkpoint: {checkpoint_path}", flush=True)
        grid = Grid.load_checkpoint(checkpoint_path)
        start_gen = len(grid.log)
    else:
        rng = np.random.default_rng(seed)
        grid = Grid(width=width, height=height, b=b, mu=mu, lam=lam, rng=rng)
        grid.snapshots = {}
        start_gen = 0
        if 0 in snap_set:
            grid.snapshots[0] = grid.strategy_grid()

    if start_gen >= num_generations:
        return grid

    iterator = tqdm(
        range(start_gen, num_generations),
        desc=f"μ={mu} λ={lam} b={b}",
        unit="gen",
        leave=True,
        disable=not progress,
        initial=start_gen,
        total=num_generations,
    )

    for g in iterator:
        grid.step(g)

        if g + 1 in snap_set:
            if not hasattr(grid, "snapshots"):
                grid.snapshots = {}
            grid.snapshots[g + 1] = grid.strategy_grid()

        if checkpoint_path and (g + 1) % checkpoint_every == 0:
            grid.save_checkpoint(checkpoint_path)

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    return grid


def averaged_run(
    mu=MUTATION_RATE,
    lam=LAMBDA,
    b=B,
    num_generations=NUM_GEN,
    num_seeds=NUM_SEEDS,
    base_seed=BASE_SEED,
    width=GRID_WIDTH,
    height=GRID_HEIGHT,
    progress=True,
    checkpoint_dir=None,
):
    all_grids = []

    for i in range(num_seeds):
        seed = base_seed + i * 4
        checkpoint_path = None
        if checkpoint_dir:
            prefix = _make_prefix(b, mu, lam, seed)
            checkpoint_path = os.path.join(checkpoint_dir, f"{prefix}.pkl")

        g = single_run(
            mu=mu, lam=lam, b=b,
            num_generations=num_generations,
            seed=seed, width=width, height=height,
            progress=progress,
            checkpoint_path=checkpoint_path,
        )
        all_grids.append(g)

    avg_log = _average_logs([g.log for g in all_grids])
    eq_coop = float(np.mean([g.equilibrium_coop() for g in all_grids]))
    eq_action_coop = float(np.mean([g.equilibrium_action_coop() for g in all_grids]))

    return {
        "log": avg_log,
        "eq_coop": eq_coop,
        "eq_action_coop": eq_action_coop,
        "all_grids": all_grids,
    }


def _average_logs(logs):
    num_gens = len(logs[0])
    avg = []

    for g in range(num_gens):
        entries = [log[g] for log in logs]
        gen_avg = {"generation": g}

        for field in ("coop_rate", "population", "avg_age"):
            gen_avg[field] = float(np.mean([e[field] for e in entries]))

        for field in ("counts", "fractions", "avg_payoffs"):
            gen_avg[field] = {
                s: float(np.mean([e[field][s] for e in entries]))
                for s in range(NUM)
            }

        avg.append(gen_avg)

    return avg


def _make_prefix(b, mu, lam, seed):
    b_str = f"b{b:.1f}".replace(".", "-")
    mu_str = f"mu{mu:.3f}".replace(".", "-")
    lam_str = f"lam{lam:.3f}".replace(".", "-")
    return f"{b_str}_{mu_str}_{lam_str}_seed{seed}"


def mu_lambda_sweep(
    b=B,
    mu_values=None,
    lambda_values=None,
    num_generations=NUM_GEN,
    num_seeds=NUM_SEEDS,
    base_seed=BASE_SEED,
    width=GRID_WIDTH,
    height=GRID_HEIGHT,
    progress=True,
    checkpoint_dir=CHECKPOINT_DIR,
):
    if mu_values is None: mu_values = MUTATION_VALUES
    if lambda_values is None: lambda_values = LAMBDA_VALUES

    heatmap = np.zeros((len(lambda_values), len(mu_values)))
    heatmap_action = np.zeros((len(lambda_values), len(mu_values)))
    combos = [(lam, mu) for lam in lambda_values for mu in mu_values]
    total = len(combos)

    for idx, (lam, mu) in enumerate(combos):
        i = lambda_values.index(lam)
        j = mu_values.index(mu)

        if progress:
            print(f"[{idx+1}/{total}] μ={mu} λ={lam} b={b}", flush=True)

        result = averaged_run(
            mu=mu, lam=lam, b=b,
            num_generations=num_generations,
            num_seeds=num_seeds, base_seed=base_seed,
            width=width, height=height,
            progress=progress,
            checkpoint_dir=checkpoint_dir,
        )

        heatmap[i, j] = result["eq_coop"]
        heatmap_action[i, j] = result["eq_action_coop"]

    return {
        "mu_values": mu_values,
        "lambda_values": lambda_values,
        "heatmap": heatmap,
        "heatmap_action": heatmap_action,
        "b": b,
    }


def full_sweep(
    b_values=None,
    mu_values=None,
    lambda_values=None,
    num_generations=NUM_GEN,
    num_seeds=NUM_SEEDS,
    base_seed=BASE_SEED,
    width=GRID_WIDTH,
    height=GRID_HEIGHT,
    progress=True,
    checkpoint_dir=CHECKPOINT_DIR,
):
    if b_values is None: b_values = B_VALUES
    results = []

    for b in b_values:
        if progress:
            print(f"\n=== b = {b:.1f} ===", flush=True)

        res = mu_lambda_sweep(
            b=b,
            mu_values=mu_values,
            lambda_values=lambda_values,
            num_generations=num_generations,
            num_seeds=num_seeds,
            base_seed=base_seed,
            width=width,
            height=height,
            progress=progress,
            checkpoint_dir=checkpoint_dir,
        )
        results.append(res)

    return results


def extract_time_series(log, field="fractions"):
    series = {s: [] for s in range(NUM)}

    for entry in log:
        for s in range(NUM):
            series[s].append(entry[field][s])

    return {s: np.array(v) for s, v in series.items()}
