import pandas as pd

# Load the dataset once
df = pd.read_csv("tcc_ceds_music.csv")


def _select_columns(data: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    cols = ["artist_name", "track_name", "genre", "topic"]
    cols = [c for c in cols if c in data.columns]
    return data[cols].head(limit).reset_index(drop=True)


def recommend_song(song_name: str, limit: int = 10) -> pd.DataFrame:
    """
    Very simple song-based recommendation:
    - If a name is given, filter tracks that contain that name.
    - If nothing matches (or no name), just return some songs.
    """
    data = df

    if song_name:
        data = data[data["track_name"].str.contains(song_name, case=False, na=False)]

    if data.empty:
        data = df

    return _select_columns(data, limit)


def recommend_by_mood(mood: str, limit: int = 10) -> pd.DataFrame:
    """
    Simple mood-based recommendation using a few numeric columns.
    """
    data = df
    mood = mood.lower()

    if mood == "happy":
        data = data[(data["valence"] >= 0.6) & (data["energy"] >= 0.6)]
    elif mood == "sad":
        data = data[(data["valence"] <= 0.4) | (data["topic"] == "sadness")]
    elif mood == "sleepy":
        data = data[(data["acousticness"] >= 0.5) & (data["energy"] <= 0.5)]

    if data.empty:
        data = df

    return _select_columns(data, limit)
