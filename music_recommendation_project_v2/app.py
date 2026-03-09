import streamlit as st
import pandas as pd
from model import recommend_song, recommend_by_mood


st.set_page_config(
    page_title="Music Recommendation System",
    page_icon="🎵",
    layout="centered"
)


st.title("🎵 Music Recommendation System")

st.write("Get song recommendations based on a song or your mood.")


option = st.selectbox(
    "Choose Recommendation Type",
    ["Song Based Recommendation", "Mood Based Recommendation"]
)


# SONG BASED RECOMMENDATION
if option == "Song Based Recommendation":

    st.subheader("🎧 Recommend Songs Similar to a Song")

    song_name = st.text_input("Enter Song Name")

    if st.button("Recommend"):

        if song_name != "":

            result = recommend_song(song_name)

            st.write("### Recommended Songs")

            st.dataframe(result)

        else:
            st.warning("Please enter a song name.")


# MOOD BASED RECOMMENDATION
elif option == "Mood Based Recommendation":

    st.subheader("😊 Recommend Songs Based on Mood")

    mood = st.selectbox(
        "Select Mood",
        ["Happy", "Sad", "Sleepy"]
    )

    if st.button("Recommend Songs"):

        result = recommend_by_mood(mood)

        st.write("### Songs For Your Mood")

        st.dataframe(result)