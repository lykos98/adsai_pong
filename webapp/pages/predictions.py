import streamlit as st
import pandas as pd
import os
import sys
sys.path.append('model')
from main import update

st.set_page_config(
        page_title="Predictions",
)

if 'win_probability_matrix.csv' not in os.listdir('model/results'):
    update()
    
win_probability_matrix = pd.read_csv('model/results/win_probability_matrix.csv', index_col=0)
latent_skill = pd.read_csv('model/results/skills.csv', index_col=0)
matchup = pd.read_csv('model/results/match_up_bonus_matrix.csv', index_col=0)

st.title("Model")

st.subheader("Win probability")
st.dataframe(win_probability_matrix, use_container_width=True)

st.subheader("Latent skill")
st.dataframe(latent_skill, use_container_width=True)

st.subheader("Matchup effect")
st.dataframe(matchup, use_container_width=True)