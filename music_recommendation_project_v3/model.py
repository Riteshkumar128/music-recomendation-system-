import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
import numpy as np

full_data = pd.read_csv("tcc_ceds_music.csv")

# Keep all columns for comprehensive analysis
data = full_data.copy()
data = data.dropna()

# Create combined text features for TF-IDF
data["text_features"] = (data["track_name"].astype(str) + " " + 
                         data["artist_name"].astype(str) + " " + 
                         data["genre"].astype(str) + " " +
                         data["lyrics"].astype(str) + " " +
                         data["topic"].astype(str))

# Apply TF-IDF on text features
tfidf = TfidfVectorizer(stop_words='english', max_features=300)
text_matrix = tfidf.fit_transform(data["text_features"])

# Extract numeric features for similarity
numeric_features = [
    'danceability', 'loudness', 'acousticness', 'instrumentalness', 
    'valence', 'energy', 'len',
    'dating', 'violence', 'world/life', 'night/time', 'shake the audience',
    'family/gospel', 'romantic', 'communication', 'obscene', 'music',
    'movement/places', 'light/visual perceptions', 'family/spiritual',
    'like/girls', 'sadness', 'feelings'
]

# Normalize numeric features
scaler = StandardScaler()
numeric_matrix = scaler.fit_transform(data[numeric_features].fillna(0))

def find_songs(song_name=None, artist_name=None, year=None):
    """Find songs by name, artist, or year"""
    results = full_data.copy()
    
    if song_name:
        results = results[results['track_name'].str.contains(song_name, case=False, na=False)]
    
    if artist_name:
        results = results[results['artist_name'].str.contains(artist_name, case=False, na=False)]
    
    if year:
        results = results[results['release_date'].astype(str).str.contains(str(year), na=False)]
    
    return results[['track_name','artist_name','release_date','genre']].head(20)

def recommend(song, feature=None):
    """Get recommendations based on comprehensive similarity and sort by feature"""
    try:
        index = data[data['track_name'].str.lower()==song.lower()].index[0]
    except (IndexError, ValueError):
        return None

    # Calculate text similarity for this song only
    text_similarity = cosine_similarity(text_matrix[index], text_matrix)[0]
    
    # Calculate numeric similarity for this song only
    numeric_similarity = cosine_similarity([numeric_matrix[index]], numeric_matrix)[0]
    
    # Weighted combination
    combined_similarity = 0.4 * text_similarity + 0.6 * numeric_similarity
    
    # Get top recommendations (excluding the song itself)
    scores = list(enumerate(combined_similarity))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:20]
    song_index = [i[0] for i in scores]
    
    result = full_data.iloc[song_index].copy()
    
    # Sort by feature if specified
    if feature is not None and feature != "None":
        if feature in result.columns:
            result = result.sort_values(by=feature, ascending=False)
    
    # Show track_name, artist_name, release_date, and lyrics
    display_columns = ['track_name','artist_name','release_date','lyrics']
    
    # Only include columns that exist
    display_columns = [col for col in display_columns if col in result.columns]
    
    return result[display_columns].reset_index(drop=True)

def get_songs_by_feature(feature=None):
    """Get top songs sorted by selected characteristic/feature"""
    if feature is None or feature == "None":
        # Return random top songs if no feature selected
        result = full_data.sample(n=min(20, len(full_data))).copy()
    else:
        if feature in full_data.columns:
            # Sort by the selected feature in descending order
            result = full_data.nlargest(20, feature).copy()
        else:
            return None
    
    # Show track_name, artist_name, release_date, and lyrics
    display_columns = ['track_name','artist_name','release_date','lyrics']
    
    # Only include columns that exist
    display_columns = [col for col in display_columns if col in result.columns]
    
    return result[display_columns].reset_index(drop=True)