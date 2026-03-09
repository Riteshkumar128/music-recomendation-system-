import streamlit as st
from model import recommend
import pandas as pd

st.title("Music Recommendation System")

song = st.text_input("Enter Song Name")

if st.button("Recommend"):

    result = recommend(song)

    st.write(result)