import sqlite3
import pandas as pd

conn = sqlite3.connect('pingpong.db')
c = conn.cursor()

c.execute('''CREATE TABLE IF NOT EXISTS players (player_id INTEGER PRIMARY KEY, 
                                                 nickname TEXT(12) UNIQUE, 
                                                 name     TEXT(12),
                                                 surname  TEXT(24),
                                                 bio      TEXT(1000), 
                                                 elo      FLOAT)''')
c.execute('''CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY,
                player1_id INTEGER,
                player2_id INTEGER,
                winner_id INTEGER,
                date DATE,
                elo_player1 FLOAT,
                elo_player2 FLOAT,
                delta_elo_player1 FLOAT,
                delta_elo_player2 FLOAT,
                FOREIGN KEY (player1_id) REFERENCES players(player_id),
                FOREIGN KEY (player2_id) REFERENCES players(player_id),
                FOREIGN KEY (winner_id) REFERENCES players(player_id)
            )''')

try:
    c.execute(''' INSERT INTO players (nickname,name,surname,bio,elo) 
                        VALUES (?, ?, ?, ?, ?)''', (
                            "nope", "Fra", "Tom","uhuh",400
                            ) )
except:
    print("Nope")
conn.commit()

c.execute("SELECT * FROM players")
for q in c.fetchall():
    print(q)





