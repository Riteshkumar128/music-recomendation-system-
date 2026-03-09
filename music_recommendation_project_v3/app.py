import streamlit as st
import pandas as pd
from model import recommend, find_songs, get_songs_by_feature

# Page configuration
st.set_page_config(page_title="Music Recommendation System", page_icon="🎵", layout="centered")

# Initialize session state
if 'search_results' not in st.session_state:
    st.session_state.search_results = None
if 'selected_song' not in st.session_state:
    st.session_state.selected_song = None

# Title
st.markdown("<h1 style='text-align: center; color: #FF4B4B;'>Music Recommendation System</h1>", unsafe_allow_html=True)

st.write("Find songs and get personalized recommendations based on music characteristics.")

st.write("---")

# SECTION 1: Find Songs
st.subheader("🔍 Find Songs by Name, Artist, or Year")

col1, col2, col3 = st.columns(3)

with col1:
    search_song = st.text_input("🎧 Song Name")

with col2:
    search_artist = st.text_input("👤 Artist Name")

with col3:
    search_year = st.text_input("📅 Year (e.g., 1950)")

if st.button("🔎 Search Songs"):
    search_results = find_songs(search_song if search_song else None, 
                               search_artist if search_artist else None,
                               search_year if search_year else None)
    st.session_state.search_results = search_results
    
    if len(search_results) > 0:
        st.success(f"Found {len(search_results)} song(s)!")
        st.dataframe(search_results)
        
        # Allow user to select a song for recommendations
        st.write("**Select a song from the results above for recommendations:**")
        selected_idx = st.selectbox("Choose a song", range(len(search_results)), 
                                   format_func=lambda i: f"{search_results.iloc[i]['track_name']} - {search_results.iloc[i]['artist_name']}")
        st.session_state.selected_song = search_results.iloc[selected_idx]['track_name']
    else:
        st.warning("No songs found with those criteria.")

st.write("---")

# SECTION 2: Get Recommendations by Song Characteristics
st.subheader("🎶 Get Recommendations by Song Characteristics")

st.write("**Option 1: Get top songs by selecting a characteristic**")
rec_feature = st.selectbox("🎼 Select Feature to Recommend", [
    "None", "energy", "danceability", "valence", "acousticness", "loudness", "instrumentalness"
])

if st.button("🎵 Get Top Songs by Feature"):
    result = get_songs_by_feature(rec_feature)
    
    if result is None:
        st.error(f"Could not get recommendations for feature '{rec_feature}'")
    else:
        st.success(f"✅ Top songs by: {rec_feature.upper()}")
        st.write(f"### 🎵 Top {len(result)} Songs by {rec_feature.upper()}")
        st.dataframe(result, use_container_width=True)

st.write("---")
st.write("**Option 2: Get similar songs based on your selected song**")

if st.button("🎵 Get Similar Songs"):
    if st.session_state.selected_song is None:
        st.warning("⚠️ Please search for a song and select it from the results above.")
    else:
        result = recommend(st.session_state.selected_song, rec_feature)
        
        if result is None:
            st.error(f"Could not get recommendations for '{st.session_state.selected_song}'")
        else:
            st.success(f"✅ Similar songs to: {st.session_state.selected_song}")
            st.write(f"### 🎵 Songs Similar to '{st.session_state.selected_song}'")
            st.dataframe(result, use_container_width=True)

st.write("---")

# Footer
st.markdown(
"""
<center>
Developed using Machine Learning & Streamlit  
</center>
""",
unsafe_allow_html=True
)