import sqlite3

import numpy as np
import pandas as pd


def load_database(path: str) -> dict[str, pd.DataFrame]:
    # Connect to the SQLite database
    conn = sqlite3.connect(path)

    # Query to get all table names
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    tables = pd.read_sql_query(query, conn)

    # Dictionary to store DataFrames
    dataframes = {}

    # Iterate over each table and load it into a DataFrame
    for table_name in tables["name"]:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        dataframes[table_name] = df

    # Close the connection
    conn.close()

    return dataframes


def get_players_and_games(
    dfs: dict[str, pd.DataFrame]
) -> tuple[pd.Series, pd.DataFrame]:
    # Get the list of players
    players = dfs["players"][["player_id", "nickname"]].copy()
    games = dfs["matches"][["winner_id", "loser_id"]]

    players["id"] = np.arange(len(players))
    players.set_index("player_id", inplace=True)

    # Substitute the player_id with the nickname
    winners = players["nickname"].loc[games["winner_id"]].values
    losers = players["nickname"].loc[games["loser_id"]].values
    games = pd.DataFrame({"winner": winners, "loser": losers})
    players.set_index("id", inplace=True)
    players = pd.Series(players["nickname"].values, name="player")

    return players, games
