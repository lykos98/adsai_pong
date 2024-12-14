import streamlit as st
import pandas as pd
import os
import sys
sys.path.append('model')
from main import update, there_are_new_games


def show_predictions():
    win_probability_matrix = pd.read_csv('model/results/win_probability_matrix.csv', index_col=0)
    latent_skill = pd.read_csv('model/results/skills.csv', index_col=0)
    matchup = pd.read_csv('model/results/match_up_bonus_matrix.csv', index_col=0)

    st.title("Predictions")

    st.subheader("Win probability")
    st.dataframe(win_probability_matrix, use_container_width=True)

    st.subheader("Latent skill")
    st.dataframe(latent_skill, use_container_width=True)

    st.subheader("Matchup effect")
    st.dataframe(matchup, use_container_width=True)

st.set_page_config(
        page_title="Predictions",
)

# If no prediction has been made yet, update the model before showing anything
if (not os.path.exists('model/results')) or ('win_probability_matrix.csv' not in os.listdir('model/results')):
    update()
        
show_predictions()

# If the games have been updated, update the model and show the updated predictions (in the meanwhile, show the old ones)
if there_are_new_games():
    update()
    show_predictions()