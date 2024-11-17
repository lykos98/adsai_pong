import streamlit as st
import sqlite3
from datetime import datetime
import numpy as np
import pandas as pd
from hashlib import sha256

st.set_page_config(
        page_title="Players",
)

conn = sqlite3.connect('db/pingpong.db', check_same_thread=False)
c = conn.cursor()

def query_players():
    c.execute("""
              SELECT * FROM players 
              ORDER BY nickname ASC""")
    return c.fetchall()

@st.dialog(title="Delete player")
def delete_player(id):
    pswd = st.text_input(label = "Insert password", type = "password")
    pswd = sha256(pswd.encode('utf-8')).hexdigest()
    deleted = False

    print(id)

    try:
        c.execute('''SELECT pswd FROM players WHERE player_id = ?''', (id,))
    except:
        st.error('Cannot find player id')
        
    pswd_player = c.fetchone()[0]

    print(pswd_player, pswd )

    if st.button(label = "Delete"):
        if pswd == pswd_player:
            try:
                c.execute(''' DELETE FROM players WHERE player_id = ?''', (id,))
                conn.commit()
                deleted = True
            except:
                st.error("Wrong player id or password")
        else:
            st.error("Wrong password")
    if deleted:
        st.rerun()

col1, col2 = st.columns(2)

st.header("Insert new player")
with st.form("player"):
    nickname = st.text_input(label="Nickname", max_chars=12)
    name     = st.text_input(label="Name", max_chars=12)
    surname  = st.text_input(label="Surname", max_chars=24)
    bio      = st.text_input(label="Bio", max_chars=140)
    password = st.text_input(label="Password", type="password")
    password_confirm = st.text_input(label="Confirm Password", type="password")
    inserted = False

    if st.form_submit_button():
        if len(password) >= 4:
            if sha256(password.encode('utf-8')).hexdigest() != sha256(password_confirm.encode('utf-8')).hexdigest():
                st.warning("Passwords do not match")
            else:
                try:
                    c.execute(''' INSERT INTO players (nickname,name,surname,bio,elo,wins,losses,pswd) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', 
                              (nickname, 
                               name, 
                               surname,
                               bio,
                               400, 
                               0,
                               0,
                               sha256(password.encode('utf-8')).hexdigest())) 
                    conn.commit()
                    inserted = True
                except:
                    st.warning("Cannot insert a player with the same nickname")
        else:
            st.warning("Password should be at least 4 characters")
    if inserted:
        st.rerun()

st.header("Player list")

results = query_players()
if len(results) > 0:
    for r in results:
        col1, col2 = st.columns([3,1])
        (id, nick, name, surname, bio, elo, wins, losses, hash) = r

        wins   = int(wins)
        losses = int(losses)
        if (wins + losses) == 0:
            winrate = 0.
        else:
            winrate = wins / (wins + losses)

        if winrate > 0.5:
            winrate_col = "green"
        else:
            winrate_col = "red"

        with col1:
            st.markdown(f"##### :blue[{nick}] (*{name} {surname}*)")
            
            st.markdown(f"**Elo** :blue-background[{elo:.1f}] **win rate** :{winrate_col}-background[{winrate:.2f}]" ) 
        with col2:
            if st.button(key = id, label = "Delete"):
                delete_player(id)

        st.divider()
else:
    st.write("No players yet inserted")




