import os
import pickle
import numpy as np
from tqdm import tqdm
from config import (
    GRID_WIDTH, GRID_HEIGHT, INITIAL_POPULATION_DENSITY,
    INITIAL_STRATEGY_PROBABILITIES, R, P, S, B,
    REPRODUCTION_RATE, MUTATION_RATE, LAMBDA, NUM,
    ALLC, TFT, PAVLOV, GRUDGER, NUM_GEN, EQUILIBRIUM_WINDOW,
)
from strategies import get_action, COOPERATE, DEFECT


class Agent:
    def __init__(self, uid, strategy, age=0):
        self.uid = uid
        self.strategy = strategy
        self.age = age
        self.payoff = 0
        self.last_payoff = None
        self.played = False
        self.grudge_set = set()


    def reset_payoff(self):
        if self.played:
            self.last_payoff = self.payoff
        self.payoff = 0
        self.played = False


class Grid:
    def __init__(
        self,
        width=GRID_WIDTH,
        height=GRID_HEIGHT,
        initial_density=INITIAL_POPULATION_DENSITY,
        strategy_fractions=None,
        b=B,
        mu=MUTATION_RATE,
        lam=LAMBDA,
        p_repro=REPRODUCTION_RATE,
        rng=None,
    ):
        self.W = width
        self.H = height
        self.b = b
        self.mu = mu
        self.lam = lam
        self.p_repro = p_repro
        self.rng = rng if rng is not None else np.random.default_rng()

        if strategy_fractions is None:
            strategy_fractions = INITIAL_STRATEGY_PROBABILITIES
        self.strategy_fractions = strategy_fractions

        self.cells = np.full((self.H, self.W), None, dtype=object)

        self._next_uid = 0

        self.agent_positions: dict = {}

        self.history: dict = {}
        self.history_index: dict = {}

        self.grudge_against: dict = {}

        self.log = []
        self._initialize_population(initial_density)


    def _make_agent(self, strategy, age=0):
        self._next_uid += 1
        return Agent(uid=self._next_uid, strategy=strategy, age=age)


    def _place(self, r, c, agent):
        self.cells[r, c] = agent
        self.agent_positions[agent.uid] = (r, c)


    def _initialize_population(self, density):
        strats = np.arange(NUM)
        probs = np.array(self.strategy_fractions, dtype=float)
        probs /= probs.sum()

        for r in range(self.H):
            for c in range(self.W):
                if self.rng.random() < density:
                    s = int(self.rng.choice(strats, p=probs))
                    self._place(r, c, self._make_agent(strategy=s))


    def _neighbours(self, row, col):
        candidates = (
            ((row - 1) % self.H, col),
            ((row + 1) % self.H, col),
            (row, (col - 1) % self.W),
            (row, (col + 1) % self.W),
        )
        return [(r, c) for r, c in candidates if self.cells[r, c] is not None]


    def _empty_neighbours(self, row, col):
        candidates = (
            ((row - 1) % self.H, col),
            ((row + 1) % self.H, col),
            (row, (col - 1) % self.W),
            (row, (col + 1) % self.W),
        )
        return [(r, c) for r, c in candidates if self.cells[r, c] is None]


    def _payoff(self, my_action, nb_action):
        T = self.b
        if my_action == COOPERATE and nb_action == COOPERATE:
            return R
        if my_action == COOPERATE and nb_action == DEFECT:
            return S
        if my_action == DEFECT and nb_action == COOPERATE:
            return T
        return P


    def _set_history(self, key, action):
        if key not in self.history:
            a, b = key
            self.history_index.setdefault(a, set()).add(key)
            self.history_index.setdefault(b, set()).add(key)
        self.history[key] = action


    def _clear_history_for(self, uid):
        keys = self.history_index.pop(uid, ())
        for key in keys:
            self.history.pop(key, None)
            a, b = key
            other = a if b == uid else b
            other_set = self.history_index.get(other)
            if other_set is not None:
                other_set.discard(key)


    def _add_grudge(self, grudger, opponent_uid):
        if opponent_uid not in grudger.grudge_set:
            grudger.grudge_set.add(opponent_uid)
            self.grudge_against.setdefault(opponent_uid, set()).add(grudger.uid)


    def _drop_dead_uid_from_grudges(self, dead_uid):
        for grudger_uid in self.grudge_against.pop(dead_uid, ()):
            pos = self.agent_positions.get(grudger_uid)
            if pos is None:
                continue
            grudger = self.cells[pos]
            if grudger is not None:
                grudger.grudge_set.discard(dead_uid)


    def _drop_grudger_from_index(self, dying_grudger):
        for opp_uid in dying_grudger.grudge_set:
            s = self.grudge_against.get(opp_uid)
            if s is None:
                continue
            s.discard(dying_grudger.uid)
            if not s:
                del self.grudge_against[opp_uid]


    def _step_interact(self):
        action_log = {}
        for r in range(self.H):
            for c in range(self.W):
                agent = self.cells[r, c]
                if agent is None:
                    continue

                agent.reset_payoff()
                for (nr, nc) in self._neighbours(r, c):
                    nb = self.cells[nr, nc]
                    key = (agent.uid, nb.uid)
                    if key not in action_log:
                        action_log[key] = get_action(agent, nb.uid, self.history, self.rng)

        visited = set()
        for r in range(self.H):
            for c in range(self.W):
                agent = self.cells[r, c]
                if agent is None:
                    continue
                for (nr, nc) in self._neighbours(r, c):
                    nb = self.cells[nr, nc]
                    pair = (agent.uid, nb.uid) if agent.uid < nb.uid else (nb.uid, agent.uid)
                    if pair in visited:
                        continue
                    visited.add(pair)

                    my_act = action_log[(agent.uid, nb.uid)]
                    nb_act = action_log[(nb.uid, agent.uid)]

                    agent.payoff += self._payoff(my_act, nb_act)
                    nb.payoff    += self._payoff(nb_act, my_act)
                    agent.played = True
                    nb.played = True

        return action_log


    def _step_reproduce(self):
        eligible = []

        for r in range(self.H):
            for c in range(self.W):
                agent = self.cells[r, c]
                if agent is None:
                    continue

                nb_coords = self._neighbours(r, c)
                if not nb_coords:
                    continue

                nb_payoffs = [self.cells[nr, nc].payoff for (nr, nc) in nb_coords]
                if agent.payoff > np.mean(nb_payoffs):
                    eligible.append((r, c))

        self.rng.shuffle(eligible)

        for (r, c) in eligible:
            agent = self.cells[r, c]
            if self.rng.random() > self.p_repro:
                continue

            empty = self._empty_neighbours(r, c)
            if not empty:
                continue

            nr, nc = empty[self.rng.integers(len(empty))]
            strategy = agent.strategy
            if self.rng.random() < self.mu:
                others = [s for s in range(NUM) if s != strategy]
                strategy = int(self.rng.choice(others))

            self._place(nr, nc, self._make_agent(strategy=strategy, age=0))


    def _step_age_and_die(self):
        for r in range(self.H):
            for c in range(self.W):
                agent = self.cells[r, c]
                if agent is None:
                    continue

                agent.age += 1
                p_death = 1.0 - np.exp(-self.lam * agent.age)
                if self.rng.random() < p_death:
                    self._kill(r, c, agent)


    def _kill(self, r, c, agent):
        if agent.strategy == GRUDGER and agent.grudge_set:
            self._drop_grudger_from_index(agent)
        self._clear_history_for(agent.uid)
        self._drop_dead_uid_from_grudges(agent.uid)
        self.agent_positions.pop(agent.uid, None)
        self.cells[r, c] = None


    def _step_update_history(self, action_log):
        for (agent_uid, nb_uid), action in action_log.items():
            self._set_history((nb_uid, agent_uid), action)

        for r in range(self.H):
            for c in range(self.W):
                agent = self.cells[r, c]
                if agent is None or agent.strategy != GRUDGER:
                    continue

                for (nr, nc) in self._neighbours(r, c):
                    nb = self.cells[nr, nc]
                    nb_action = action_log.get((nb.uid, agent.uid))
                    if nb_action == DEFECT:
                        self._add_grudge(agent, nb.uid)


    def _collect_metrics(self, generation, action_log):
        counts = {s: 0 for s in range(NUM)}
        payoffs = {s: [] for s in range(NUM)}
        ages = []
        total = 0
        coop_actions = 0
        total_actions = 0

        for r in range(self.H):
            for c in range(self.W):
                agent = self.cells[r, c]
                if agent is None:
                    continue

                counts[agent.strategy] += 1
                payoffs[agent.strategy].append(agent.payoff)
                ages.append(agent.age)
                total += 1

        for action in action_log.values():
            total_actions += 1
            if action == COOPERATE:
                coop_actions += 1

        fractions = {s: counts[s] / total if total > 0 else 0 for s in range(NUM)}
        avg_payoffs = {s: float(np.mean(payoffs[s])) if payoffs[s] else 0 for s in range(NUM)}
        coop_rate = coop_actions / total_actions if total_actions > 0 else 0

        self.log.append({
            "generation": generation,
            "counts": counts,
            "fractions": fractions,
            "avg_payoffs": avg_payoffs,
            "coop_rate": coop_rate,
            "population": total,
            "avg_age": float(np.mean(ages)) if ages else 0,
        })


    def step(self, generation):
        action_log = self._step_interact()
        self._step_reproduce()
        self._step_update_history(action_log)
        self._step_age_and_die()
        self._collect_metrics(generation, action_log)


    def run(self, num_generations=None, progress=False):
        if num_generations is None:
            num_generations = NUM_GEN

        iterator = range(num_generations)
        if progress:
            iterator = tqdm(iterator, desc="Simulating", unit="gen")

        for g in iterator:
            self.step(g)


    def strategy_grid(self):
        grid = np.full((self.H, self.W), -1, dtype=int)

        for r in range(self.H):
            for c in range(self.W):
                if self.cells[r, c] is not None:
                    grid[r, c] = self.cells[r, c].strategy

        return grid


    def equilibrium_coop(self, window=None):
        if window is None:
            window = EQUILIBRIUM_WINDOW

        recent = self.log[-window:]
        if not recent:
            return 0

        coop_strats = {ALLC, TFT, PAVLOV, GRUDGER}
        coop_fracs = []

        for entry in recent:
            total = entry["population"]
            if total == 0:
                continue

            coop_count = sum(entry["counts"][s] for s in coop_strats)
            coop_fracs.append(coop_count / total)

        return float(np.mean(coop_fracs)) if coop_fracs else 0


    def equilibrium_action_coop(self, window=None):
        if window is None:
            window = EQUILIBRIUM_WINDOW

        recent = self.log[-window:]
        if not recent:
            return 0

        rates = [entry["coop_rate"] for entry in recent if entry["population"] > 0]
        return float(np.mean(rates)) if rates else 0


    def save_checkpoint(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load_checkpoint(path):
        with open(path, "rb") as f:
            return pickle.load(f)
