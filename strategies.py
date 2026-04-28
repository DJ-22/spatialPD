from config import ALLC, ALLD, TFT, PAVLOV, GRUDGER, RANDOM, R


COOPERATE = 1
DEFECT = 0


def action_allc(agent, neighbour_id, history, rng):
    return COOPERATE


def action_alld(agent, neighbour_id, history, rng):
    return DEFECT


def action_tft(agent, neighbour_id, history, rng):
    return history.get((agent.uid, neighbour_id), COOPERATE)


def action_pavlov(agent, neighbour_id, history, rng):
    if agent.last_payoff is None:
        return COOPERATE

    return COOPERATE if agent.last_payoff >= R else DEFECT


def action_grudger(agent, neighbour_id, history, rng):
    if neighbour_id in agent.grudge_set:
        return DEFECT

    return COOPERATE


def action_random(agent, neighbour_id, history, rng):
    return rng.choice([COOPERATE, DEFECT])


ACTION_FN = {
    ALLC: action_allc,
    ALLD: action_alld,
    TFT: action_tft,
    PAVLOV: action_pavlov,
    GRUDGER: action_grudger,
    RANDOM: action_random,
}


def get_action(agent, neighbour_id, history, rng):
    fn = ACTION_FN.get(agent.strategy)

    if fn is None:
        raise ValueError(f"Invalid strategy: {agent.strategy}")

    return fn(agent, neighbour_id, history, rng)
