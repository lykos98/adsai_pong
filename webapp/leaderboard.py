import streamlit as st
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from hashlib import sha256
import os

# Connect to SQLite database

conn = sqlite3.connect('db/pingpong.db', check_same_thread=False)
conn.row_factory = sqlite3.Row
c = conn.cursor()


def query_players():
    c.execute("""
              SELECT * FROM players
              ORDER BY elo DESC
              """)
    return c.fetchall()

c.execute('''CREATE TABLE IF NOT EXISTS players (player_id INTEGER PRIMARY KEY, 
                                                 nickname TEXT(12) UNIQUE, 
                                                 name     TEXT(12),
                                                 surname  TEXT(24),
                                                 bio      TEXT(1000), 
                                                 elo      FLOAT,
                                                 wins     INTEGER,
                                                 losses   INTEGER,
                                                 pswd     CHAR(64))''')
c.execute('''CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                winner_id INTEGER,
                loser_id INTEGER,
                date DATETIME,
                elo_winner FLOAT,
                elo_loser FLOAT,
                delta_elo_winner FLOAT,
                delta_elo_loser FLOAT,
                FOREIGN KEY (loser_id) REFERENCES players(player_id),
                FOREIGN KEY (winner_id) REFERENCES players(player_id)
            )''')

st.set_page_config(
        page_title="ADSAI pong",
)

st.title("Leaderboard")


results = query_players()
if len(results) > 0:
    for i,r in enumerate(results):

        wins   = int(r['wins'])
        losses = int(r['losses'])
        if (wins + losses) == 0:
            winrate = 0.
        else:
            winrate = wins / (wins + losses)

        if winrate > 0.5:
            winrate_col = "green"
        else:
            winrate_col = "red"

        col1, col2, col3, col4 = st.columns([1,1,1,2], vertical_alignment="bottom")
        with col1:
            st.markdown(f"### {i + 1}°")

        with col2:
            st.markdown(f"##### :blue[{r['nickname']}]")
            
        with col3:
            st.markdown(f"#### Elo :blue-background[{r['elo']:.1f}]")

        with col4:
            st.markdown(f"#### win rate :{winrate_col}-background[{winrate:.2f}]" ) 

        st.divider()
else:
    st.write("No players yet inserted")
