import math
import os

import numpy as np
import pandas as pd
import pyro
import torch
import tqdm
from pyro.infer import SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide.guides import AutoDiagonalNormal
from pyro.optim import RAdam
import hashlib
import argparse
import random

#sys.path.append(".")

from data import get_players_and_games, load_database
from model import model, to_match_up_bonus_matrix, to_skills_array

DEBUG = False
SEED = 69

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def vi(*, model, guide, data, n_steps, num_particles, opt_params, patience: int = 100):

    optimizer = RAdam(opt_params)

    # setup the inference algorithm
    svi = SVI(
        model,
        guide,
        optimizer,
        loss=Trace_ELBO(vectorize_particles=False, num_particles=num_particles),
    )

    losses = np.zeros(n_steps)

    # Do gradient steps
    with tqdm.tqdm(range(n_steps)) as pbar:
        for i in pbar:
            losses[i] = svi.step(data)
            pbar.set_postfix(ELBO=f"{losses[i]:.2f}")
            if i > patience and losses[i] > losses[i - patience]:
                losses[i] = svi.step(
                    data
                )  # one more steps is probably going to improve the result
                break

    return guide


def get_games_tensor(games, names):
    names_map = pd.Series(names.index, index=names.values.squeeze())
    games_tensor = torch.zeros(len(games), 2)
    games_tensor[:, 0] = torch.tensor(names_map.loc[games["winner"]].values)
    games_tensor[:, 1] = torch.tensor(names_map.loc[games["loser"]].values)
    return games_tensor.long()


def store_guide(guide, path):
    torch.save(guide, path)


def store_skills_representation(
    map, skills_samples, match_up_bonus_matrix_samples, names, path
):
    """
    Stores a (#players x #players+1) matrix in csv format containing for each player the couple (median, std) of their skill and the match up bonus against each other player.
    """
    skills_std = skills_samples.std(dim=0)
    match_up_bonus_matrix_std = match_up_bonus_matrix_samples.std(dim=0)

    skills_df = pd.DataFrame(
        map["skills"].numpy(), columns=["skill"], index=names.values
    )
    skills_df["std"] = skills_std.numpy()
    skills_df.index.name = "player"
    skills_df = skills_df.round(2).sort_values("skill", ascending=False)

    match_up_bonus_matrix_df = pd.DataFrame(
        map["match_up_bonus_matrix"].numpy(), columns=names.values, index=names.values
    ).round(2)
    match_up_bonus_matrix_df_std = pd.DataFrame(
        match_up_bonus_matrix_std.numpy(), columns=names.values, index=names.values
    ).round(2)

    skills_df.to_csv(path + "skills.csv")
    match_up_bonus_matrix_df[skills_df.index].loc[skills_df.index].to_csv(
        path + "match_up_bonus_matrix.csv"
    )
    match_up_bonus_matrix_df_std[skills_df.index].loc[skills_df.index].to_csv(
        path + "match_up_bonus_matrix_std.csv"
    )
    return skills_df.index


def store_win_probability_matrix(win_probability_matrix, names, path):
    win_probability_matrix_df = pd.DataFrame(
        win_probability_matrix, columns=names.values, index=names.values
    ).round(2)
    sorted_names = (
        win_probability_matrix_df.sum(axis=1).sort_values(ascending=False).index
    )
    win_probability_matrix_df.index.name = "player"
    win_probability_matrix_df.columns.name = "player"
    win_probability_matrix_df[sorted_names].loc[sorted_names].to_csv(
        path + "win_probability_matrix.csv"
    )


def get_skills_samples(posterior_samples):
    return posterior_samples["skills"] - posterior_samples["skills"].mean(
        dim=-1, keepdim=True
    )  # / posterior_samples["skills"].std(dim=-1, keepdim=True)
    # return torch.cat([posterior_samples["skills"].sum(-1, keepdim=True), posterior_samples["skills"]], dim=1)


def get_match_up_bonus_matrix_samples(posterior_samples):
    sample_size = len(posterior_samples["match_up_matrix"])
    size = math.ceil(
        (posterior_samples["match_up_matrix"].shape[-1] * 2) ** 0.5
    )  # triangular number stuff
    tril_indices = torch.tril_indices(row=size, col=size, offset=-1)

    match_up_bonus_matrix_samples = torch.zeros((sample_size, size, size))
    match_up_bonus_matrix_samples[:, tril_indices[0], tril_indices[1]] = (
        posterior_samples["match_up_matrix"]
    )
    match_up_bonus_matrix_samples = (
        match_up_bonus_matrix_samples - match_up_bonus_matrix_samples.transpose(-1, -2)
    )

    # Copilot solution
    adjustment = match_up_bonus_matrix_samples.mean(dim=-1)
    match_up_bonus_matrix_samples -= adjustment.unsqueeze(-1)
    match_up_bonus_matrix_samples += adjustment.unsqueeze(-2)

    return match_up_bonus_matrix_samples


def compute_win_probability_matrix(posterior_samples):
    skills_samples = get_skills_samples(posterior_samples)
    match_up_bonus_matrix_samples = get_match_up_bonus_matrix_samples(posterior_samples)

    win_logit_samples = (
        skills_samples.unsqueeze(-1)
        + match_up_bonus_matrix_samples
        - skills_samples.unsqueeze(1)
    )  # * torch.exp(posterior_samples['log_k']).reshape(-1,1,1)

    win_prob_samples = torch.sigmoid(win_logit_samples)
    return win_prob_samples.mean(dim=0)


def tensor_hash(tensor: torch.Tensor):    
    hash_object = hashlib.sha256()
    hash_object.update(tensor.numpy().tobytes())    
    return hash_object.hexdigest()

def list_hash(input_list):
    list_string = str(input_list)    
    hash_object = hashlib.sha256()
    hash_object.update(list_string.encode('utf-8'))
    return hash_object.hexdigest()
    
def store_games_hash(games_tensor: torch.Tensor, folder: str):    
    with open(f"{folder}/games_hash.txt", 'w') as file:
        file.write(tensor_hash(games_tensor))
    

def new_games(games_tensor: torch.Tensor):
    try:
        with open(f"model/results/games_hash.txt", 'r') as file:
            old_hash = file.read()
    except FileNotFoundError:
        return True
    
    new_hash = tensor_hash(games_tensor)
    return new_hash != old_hash
    
def there_are_new_games(database_file: str = "db/pingpong.db"):
    try:
        with open(f"model/results/games_hash.txt", 'r') as file:
            old_hash = file.read()
    except FileNotFoundError:
        return True
    
    dfs = load_database(database_file)
    names, games = get_players_and_games(dfs)
    games_tensor = get_games_tensor(games, names)
    return new_games(games_tensor)

def store_players_hash(player_names: list, folder: str):    
    with open(f"{folder}/players_hash.txt", 'w') as file:
        file.write(list_hash(player_names))

def same_players(player_names):
    try:
        with open(f"model/results/players_hash.txt", 'r') as file:
            old_hash = file.read()
    except FileNotFoundError:
        return True
    
    new_hash = list_hash(player_names)
    return new_hash == old_hash
    
def initialize_guide(guide_to_reuse: str | None, player_names, default):
    if guide_to_reuse is not None and same_players(player_names) and os.path.exists(guide_to_reuse):
        print("Reusing guide")
        return torch.load(guide_to_reuse, weights_only=False)
    return default
    
    
def update() -> bool:
    set_seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument("--database_file", type=str, default="db/pingpong.db")
    args = parser.parse_args()

    dfs = load_database(args.database_file)
    names, games = get_players_and_games(dfs)
    games_tensor = get_games_tensor(games, names)
    
    if new_games(games_tensor): # Should also check if data is actually there
        
        delta_guide = initialize_guide(guide_to_reuse='model/results/delta_guide.pt', player_names = names, default = pyro.infer.autoguide.AutoDelta(model))
        guide = initialize_guide(guide_to_reuse='model/results/guide.pt', player_names = names, default = AutoDiagonalNormal(model))
        
        delta_guide = vi(
            model=model,
            data=games_tensor,
            n_steps=5000 if not DEBUG else 10,
            num_particles=1,
            opt_params={"lr": 0.005},
            patience=300,
            guide=delta_guide,
        )

        map_estimate = delta_guide.median()

        guide = vi(
            model=model,
            data=games_tensor,
            n_steps=2000 if not DEBUG else 10,
            num_particles=10,
            opt_params={"lr": 0.01},
            patience=300,
            guide=guide,
        )

        posterior_samples = Predictive(model, guide=guide, num_samples=5000 if not DEBUG else 10)(games_tensor)
        skills_samples = get_skills_samples(posterior_samples)
        match_up_bonus_matrix_samples = get_match_up_bonus_matrix_samples(posterior_samples)
        map_estimate["skills"] = to_skills_array(map_estimate["skills"])
        map_estimate["match_up_bonus_matrix"] = to_match_up_bonus_matrix(
            map_estimate["match_up_matrix"]
        )

        win_probability_matrix = compute_win_probability_matrix(posterior_samples)

        folder = "model/results/"
        os.makedirs(folder, exist_ok=True)
        store_guide(guide=delta_guide, path=folder + "delta_guide.pt")
        store_guide(guide=guide, path=folder + "guide.pt")
        store_skills_representation(
            map=map_estimate,
            skills_samples=skills_samples,
            match_up_bonus_matrix_samples=match_up_bonus_matrix_samples,
            names=names,
            path=folder,
        )
        store_win_probability_matrix(
            win_probability_matrix=win_probability_matrix, names=names, path=folder
        )
        store_games_hash(games_tensor, folder)
        store_players_hash(names, folder)
        return True
    else:
        print("No new games")
        return False
        
if __name__ == "__main__":
    update()