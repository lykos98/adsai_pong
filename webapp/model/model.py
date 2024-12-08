import math

import pyro
import pyro.distributions as dist
import torch

MODEL_MATCH_UP_BONUS_SCALE = False
PRIOR = {
    "loc_log_match_up_bonus_scale": -1.0,
    "scale_log_match_up_bonus_scale": 3.0,
}


def sample_lower_trianglular_matrix(*, size: int, name: str, dist: dist.Distribution):
    tril_indices = torch.tril_indices(row=size, col=size, offset=-1)
    m = torch.zeros(size, size)
    m[tril_indices[0], tril_indices[1]] = pyro.sample(
        name, dist.expand((tril_indices.shape[1],)).to_event(1)
    )
    return m


def to_match_up_bonus_matrix(values: torch.Tensor):
    size = math.ceil((values.shape[0] * 2) ** 0.5)  # triangular number stuff
    tril_indices = torch.tril_indices(row=size, col=size, offset=-1)
    match_up_bonus_matrix = torch.zeros(size, size)
    match_up_bonus_matrix[tril_indices[0], tril_indices[1]] = values
    match_up_bonus_matrix = match_up_bonus_matrix - match_up_bonus_matrix.T

    # Copilot solution
    adjustment = match_up_bonus_matrix.mean(dim=-1)
    match_up_bonus_matrix -= adjustment.unsqueeze(1)
    match_up_bonus_matrix += adjustment.unsqueeze(0)

    return match_up_bonus_matrix


def to_skills_array(values: torch.Tensor):
    return (
        values - values.mean()
    )  # / values.std() # Not the best, it would be better to have less parameters than location invariance


def sample_skills(*, size: int, name: str, dist: dist.Distribution):
    return to_skills_array(pyro.sample(name, dist.expand((size,)).to_event()))


def sample_match_up_bonus_matrix(*, size: int, name: str, dist: dist.Distribution):
    values = pyro.sample(name, dist.expand((size * (size - 1) // 2,)).to_event(1))
    return to_match_up_bonus_matrix(values=values)


def ever_played_with_matrix(games: torch.Tensor):
    number_of_players = games.max().item() + 1
    ever_played_with = torch.zeros(
        number_of_players, number_of_players, dtype=torch.bool
    )
    ever_played_with[games[:, 0], games[:, 1]] = True
    ever_played_with[games[:, 1], games[:, 0]] = True
    return ever_played_with


def model(games: torch.Tensor, obs="first_won"):

    number_of_players = games.max().item() + 1
    skill = sample_skills(
        name="skills", dist=dist.Normal(loc=0.0, scale=5.0), size=number_of_players
    )

    if MODEL_MATCH_UP_BONUS_SCALE:
        match_up_bonus_scale = torch.exp(pyro.sample(
            "log_match_up_bonus_scale",
            dist.Normal(
                loc=PRIOR["loc_log_match_up_bonus_scale"],
                scale=PRIOR["scale_log_match_up_bonus_scale"],
            ),
        ))
    else:
        match_up_bonus_scale = pyro.param(
            "log_match_up_bonus_scale",
            torch.tensor(0.01),
            constraint=dist.constraints.positive,
        )

    match_up_bonus_matrix = sample_match_up_bonus_matrix(
        name="match_up_matrix",
        size=number_of_players,
        dist=dist.Laplace(loc=0.0, scale=1.0 + match_up_bonus_scale),
    )

    winners = games[:, 0]
    losers = games[:, 1]
    win_logit = (
        pyro.deterministic(
            "win_logit",
            skill[winners] + match_up_bonus_matrix[winners, losers] - skill[losers],
        )
    )

    return pyro.sample(
        name=f"winner",
        fn=dist.Bernoulli(logits=win_logit).to_event(),
        obs=torch.ones(len(games)) if obs == "first_won" else None,
    )
