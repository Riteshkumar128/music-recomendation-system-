import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv("tcc_ceds_music.csv")

data = data[['track_name','artist_name','genre']]

data = data.dropna()

data["combined"] = data["track_name"] + " " + data["artist_name"] + " " + data["genre"]

tfidf = TfidfVectorizer(stop_words='english')

matrix = tfidf.fit_transform(data["combined"])

similarity = cosine_similarity(matrix)

def recommend(song):

    index = data[data['track_name']==song].index[0]

    scores = list(enumerate(similarity[index]))

    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:11]

    song_index = [i[0] for i in scores]

    return data.iloc[song_index][['track_name','artist_name','genre']]