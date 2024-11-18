import streamlit as st
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from hashlib import sha256
import math
from datetime import datetime

st.set_page_config(
        page_title="Matches",
)

# Connect to SQLite database
conn = sqlite3.connect('db/pingpong.db', check_same_thread=False)
conn.row_factory = sqlite3.Row

c = conn.cursor()

def get_win_probability(rating1, rating2, alpha = 400.):
    # Calculate and return the expected score
    return 1.0 / (1. + math.pow(10, (rating1 - rating2) / alpha))

def get_delta_elo(elo_winner, elo_loser, k = 30.):
    p_winner = get_win_probability(elo_winner, elo_loser)
    print(elo_winner, elo_loser, p_winner)
    p_loser  = 1. - p_winner

    delta_elo_winner =  k * p_winner
    delta_elo_loser  = -k * p_winner
    
    return delta_elo_winner, delta_elo_loser
        

@st.dialog(title="Delete player")
def delete_match(match_id, winner_id, loser_id):
    pswd = st.text_input(label = "Insert password of winner or loser", type = "password")
    pswd = sha256(pswd.encode('utf-8')).hexdigest()
    deleted = False

    c.execute("""SELECT * FROM matches WHERE match_id = ?""", (match_id,))
    match_info = c.fetchone()

    try:
        c.execute('''SELECT pswd,elo,wins FROM players WHERE player_id = ?''', (winner_id,))
        winner = c.fetchone()

        c.execute('''SELECT pswd,elo,losses FROM players WHERE player_id = ?''', (loser_id,))
        loser = c.fetchone()
    except:
        st.error('Cannot find player id')
        


    if st.button(label = "Delete"):
        if pswd == winner['pswd'] or pswd == loser['pswd']:
            c.execute(''' DELETE FROM matches WHERE match_id = ?''', (match_id,))

            deleted = True
            try:
                new_winner_elo = winner['elo'] - match_info['delta_elo_winner']
                new_loser_elo = loser['elo'] - match_info['delta_elo_loser']

                new_winner_wins  = winner['wins'] - 1
                new_loser_losses = loser['losses'] - 1
                c.execute("""UPDATE players
                          SET elo = ?, wins = ?
                          WHERE player_id = ?""", (new_winner_elo, new_winner_wins, winner_id))
                c.execute("""UPDATE players
                          SET elo = ?, losses = ?
                          WHERE player_id = ?""", (new_loser_elo, new_loser_losses, loser_id))
                conn.commit()
            except:
                st.error("Wrong password")
        else:
            st.error("Wrong password")
    if deleted:
        st.rerun()

def query_matches():
    c.execute("""
               SELECT 
                    COALESCE(pw.nickname, "***") AS winner_nickname, 
                    COALESCE(pl.nickname, "***") AS loser_nickname, 
                    matches.winner_id,
                    matches.loser_id,
                    matches.match_id,
                    matches.date,
                    matches.elo_winner,
                    matches.elo_loser,
                    matches.delta_elo_winner,
                    matches.delta_elo_loser
                FROM 
                    matches
                LEFT JOIN 
                    players AS pw ON matches.winner_id = pw.player_id
                LEFT JOIN 
                    players AS pl ON matches.loser_id = pl.player_id;""")
    return c.fetchall()



st.header("Insert new match")
with st.form("match"):
    c.execute("""SELECT nickname FROM players""")
    players = [p['nickname'] for p in c.fetchall()]

    col1, col2, col3 = st.columns([2,2,1], vertical_alignment="bottom")
    with col1:
        winner = st.selectbox("Winner", options = players)
    with col2:
        loser  = st.selectbox("Loser", options = players)
    with col3:
        inserted = False
        if st.form_submit_button("Add match"):
            if winner != loser:
                c.execute("""SELECT * FROM players WHERE nickname = (?)""", (winner,))            
                winner_properties = c.fetchone()
                c.execute("""SELECT * FROM players WHERE nickname = (?)""", (loser,))            
                loser_properties = c.fetchone()

                new_winner_wins  = winner_properties['wins'] + 1
                new_loser_losses = loser_properties['losses'] + 1

                (delta_elo_winner, delta_elo_loser) = get_delta_elo(float(winner_properties['elo']), float(loser_properties['elo']))

                try:
                    c.execute("""INSERT INTO matches 
                                 (match_id, winner_id, loser_id, date, elo_winner, elo_loser, delta_elo_winner, delta_elo_loser) 
                                 VALUES (?,?,?,?,?,?,?,?)""",
                                    (int(datetime.now().timestamp() * 1000),
                                     winner_properties['player_id'],
                                     loser_properties['player_id'],
                                     str(datetime.now()),
                                     winner_properties['elo'],
                                     loser_properties['elo'],
                                     delta_elo_winner,
                                     delta_elo_loser))
                    new_winner_elo = float(winner_properties['elo']) + delta_elo_winner
                    new_loser_elo = float(loser_properties['elo']) + delta_elo_loser
                    conn.commit()

                    c.execute("""UPDATE players
                                 SET elo = ?, wins = ?
                                 WHERE player_id = ?""", (new_winner_elo, new_winner_wins, winner_properties['player_id']))
                    c.execute("""UPDATE players
                                 SET elo = ?, losses = ?
                                 WHERE player_id = ?""", (new_loser_elo, new_loser_losses, loser_properties['player_id']))
                    conn.commit()
                    inserted = True
                except:
                    st.error("Cannot insert match!")


            else:
                st.error("Winner and loser cannot be the same person")

    if inserted:
        st.rerun()

results = query_matches()
if len(results) > 0:
    for r in results:
        col1, col2, col3, col4, col5, col6 = st.columns([2, 2, 2, 2, 2, 2])
        with col1:
            d_obj = datetime.fromisoformat(r['date'])
            st.write(f"""{str(d_obj.date())} {str(d_obj.hour)}:{str(d_obj.minute)}:{str(d_obj.second)}""")
        with col2:
            st.markdown(f"""##### :green[{r['winner_nickname']}]""")
        with col3:
            st.markdown(f"""Elo: :green-background[+ {r['delta_elo_winner']:.1f}] """)
        with col4:
            st.markdown(f"""##### :red[{r['loser_nickname']}]""")
        with col5:
            st.markdown(f"""Elo: :red-background[{r['delta_elo_loser']:.1f}] """)
        with col6:
            if st.button(key = r['match_id'], label = "Delete"):
                delete_match(r['match_id'], r['winner_id'], r['loser_id'])
        st.divider()
else:
    st.write("No matches inserted")
