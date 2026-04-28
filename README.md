# spatialPD

A configurable Python simulation of the **spatial Prisoner's Dilemma** with age-based mortality, strategy mutation, and a six-strategy pool. Think of it as a Python version of NetLogo's *Prisoner's Dilemma Basic Evolutionary* model, with several additional dynamics layered on top: per-agent ageing, stochastic death, mutation on reproduction, and a richer strategy library.

Every model parameter (grid size, mutation rate, ageing rate, temptation payoff, reproduction probability, initial population, initial strategy mix, …) is exposed as a CLI flag *and* a value in `config.py`. You can run a single configuration, sweep one parameter, sweep two parameters jointly, or run a full 3-axis grid sweep with averaging across multiple seeds.

## Model overview

- 2D toroidal lattice. Each cell is empty or holds one agent.
- Each agent carries a strategy, an age, and a per-generation payoff.
- Each generation: agents play PD with their **von Neumann neighbours** (top, bottom, left, right), payoff-eligible agents reproduce into empty adjacent cells (with mutation probability μ), per-pair history updates, then everyone ages and dies stochastically with `p_death(a) = 1 − exp(−λ · a)`.

### Strategies

| Strategy | Behaviour |
|---|---|
| **AllC** | Always cooperate |
| **AllD** | Always defect |
| **TfT** | Cooperate first, then mirror neighbour's last move (per pair) |
| **Pavlov** | Aggregate win-stay, lose-shift: cooperate iff total payoff ≥ R last round |
| **Grudger** | Cooperate until a neighbour defects, then defect against that neighbour forever |
| **Random** | Coin-flip cooperate/defect each pair each round |

Strategies are defined in `strategies.py` as plain functions; adding a new one is a few lines (see [Extending](#extending)).

### Compared to NetLogo's *PD Basic Evolutionary*

| | NetLogo basic | spatialPD |
|---|---|---|
| Strategy pool | C/D with simple imitation | 6 explicit strategies including memory-based (TfT, Grudger, Pavlov) |
| Reproduction | Imitation of best neighbour | Payoff-driven reproduction into empty adjacent cells |
| Mortality | None | Age-based stochastic death `1 − exp(−λa)` |
| Mutation | None | Per-offspring mutation rate μ |
| Population | Fixed (every cell occupied) | Variable (cells go empty on death) |
| Memory | None | Per-pair history for TfT, per-opponent grudge sets for Grudger |
| Sweep tooling | Manual via BehaviorSpace | Built-in CLI sweeps with checkpointing & multi-seed averaging |

## Install

```bash
pip install -r requirements.txt
```

Dependencies: `numpy`, `matplotlib`, `seaborn`, `tqdm`.

On Windows, prepend `PYTHONIOENCODING=utf-8` (bash) or set `$env:PYTHONIOENCODING="utf-8"` (PowerShell) so μ/λ characters print correctly.

## Quick start

```bash
# One simulation at default parameters
python main.py --mode single

# One simulation with custom parameters
python main.py --mode single --b 1.8 --mu 0.005 --lam 0.01 --width 50 --height 50

# Average across 5 seeds at a fixed configuration
python main.py --mode averaged --num_seeds 5

# Sweep μ × λ at fixed b, averaging across 3 seeds per cell
python main.py --mode sweep --b 1.5

# Full sweep over (b, μ, λ) — produces a heatmap per b value
python main.py --mode full_sweep
```

Each invocation creates a fresh timestamped folder under `output/`, so prior results are never overwritten. Long sweeps checkpoint every 100 generations to `checkpoints/` and auto-resume if interrupted.

## CLI modes

| Mode | What it does |
|---|---|
| `single` | One simulation at the given (b, μ, λ, seed). Saves strategy timeseries, cooperation-rate plot, and spatial snapshot grid. |
| `averaged` | Same configuration, averaged across N seeds. Saves averaged timeseries and cooperation-rate. |
| `sweep` | μ × λ heatmap at fixed b, averaged across N seeds per cell. |
| `full_sweep` | (b, μ, λ) grid sweep. Produces individual heatmaps per b plus a combined figure. |
| `compare` | Pre-defined comparison: low μ × low λ vs high μ × high λ vs intermediate, in one figure. |

## Configurable parameters

Defaults live in `config.py`. CLI flags override the defaults for a single invocation; edit `config.py` if you want a new permanent default.

| Parameter | Flag | `config.py` | Description |
|---|---|---|---|
| Grid width | `--width` | `GRID_WIDTH` | Cells in x direction (toroidal) |
| Grid height | `--height` | `GRID_HEIGHT` | Cells in y direction (toroidal) |
| Generations | `--generations` | `NUM_GEN` | How long to run |
| Equilibrium window | — | `EQUILIBRIUM_WINDOW` | Final N gens averaged for equilibrium stats |
| Initial occupancy | — | `INITIAL_POPULATION_DENSITY` | Fraction of cells initially occupied |
| Initial strategy mix | — | `INITIAL_STRATEGY_PROBABILITIES` | List of 6 probabilities (must sum to 1) |
| Reward / Punishment / Sucker | — | `R`, `P`, `S` | Standard PD payoffs |
| Temptation | `--b` | `B` | T = b in the PD payoff matrix |
| Reproduction probability | — | `REPRODUCTION_RATE` | Probability that an eligible agent actually reproduces |
| Mutation rate | `--mu` | `MUTATION_RATE` | Probability that an offspring switches strategy |
| Aging rate | `--lam` | `LAMBDA` | Death-curve steepness |
| Sweep grid: b | — | `B_VALUES` | Values of b in `full_sweep` |
| Sweep grid: μ | — | `MUTATION_VALUES` | Values of μ in sweeps |
| Sweep grid: λ | — | `LAMBDA_VALUES` | Values of λ in sweeps |
| Seeds per cell | `--num_seeds` | `NUM_SEEDS` | Random-seed replicates per parameter cell in sweeps |
| Base random seed | `--seed` | `BASE_SEED` | Reproducibility anchor |
| Output subfolder | `--output_dir` | — | Custom subfolder name (default: timestamped) |

## Outputs

For each run mode, plots land in `output/run_<timestamp>_<mode>/`:

- **`*_timeseries.png`** — strategy fractions over generations, one line per strategy.
- **`*_coop_rate.png`** — fraction of cooperative actions emitted per generation.
- **`*_snapshots.png`** — spatial colour-coded grid at gen 0, 500, 1000, and 2000 (or final).
- **`heatmap_b<b>_strategy.png`** — equilibrium fraction of cooperative-strategy agents (AllC + TfT + Pavlov + Grudger).
- **`heatmap_b<b>_action.png`** — equilibrium fraction of cooperative *actions* emitted (counts Random's 50% C contribution).
- **`heatmaps_all_b_*.png`** — combined figure across all b values.

The two cooperation metrics measure different things:

- **Strategy-based cooperation** treats *who you are* (cooperative-leaning strategy) as the unit.
- **Action-based cooperation** treats *what you actually did* (the move) as the unit. Captures Random's 50% C contribution and any defection emitted by retaliating cooperators.

The gap between them is a useful diagnostic for Random-dominated regimes.

## Repository layout

```
spatialPD/
├── simulation.py      # Grid + Agent classes, lifecycle implementation
├── strategies.py      # The six strategy action functions
├── experiment.py      # single_run, averaged_run, sweep, full_sweep
├── visualize.py       # Plot helpers (timeseries, heatmaps, snapshots)
├── config.py          # All default parameters and sweep grids
├── main.py            # CLI entry point
├── requirements.txt
└── output/            # Per-invocation timestamped result folders
```

## Extending

### Add a new strategy

1. Add a constant ID and label in `config.py` (`STRATEGY_MAPPING`, `STRATEGY_COLORS`, `NUM`).
2. Write a function in `strategies.py` with signature `(agent, neighbour_id, history, rng) -> COOPERATE | DEFECT`.
3. Register it in the `ACTION_FN` dict in `strategies.py`.
4. Add an initial-fraction entry in `INITIAL_STRATEGY_PROBABILITIES` (must still sum to 1).

### Change the payoff matrix

Edit `R`, `P`, `S` in `config.py`. The temptation `T` is set per-run from `B` (`b` in CLI).

### Change the neighbourhood

The neighbour lookup is in `simulation.py::Grid._neighbours`. Replace with Moore (8-neighbour), hex, or custom topology by editing that one method.

### Change the lifecycle order

`simulation.py::Grid.step` runs interact → reproduce → update history → age and die. Reorder there if you want a different lifecycle.

## Reproducibility

- All randomness flows through a `numpy.random.Generator` seeded from `--seed`.
- Sweep modes derive per-seed offsets deterministically (`base_seed + i * 4`).
- Long runs checkpoint every 100 generations; pickled `Grid` snapshots are saved to `checkpoints/<prefix>.pkl` and auto-loaded on re-run if present.

## License

[MIT](LICENSE) © 2026 Daksh Jain.
