import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("SpotifyFeatures.csv")

df.dropna(how="all", inplace=True)
df.drop_duplicates(subset="track_id", inplace=True)
df.reset_index(drop=True, inplace=True)


# -----------------------------
# Features
# -----------------------------
audio_features = [
    "acousticness",
    "danceability",
    "energy",
    "instrumentalness",
    "liveness",
    "loudness",
    "speechiness",
    "tempo",
    "valence"
]


# -----------------------------
# Mood Labeling (same as your code)
# -----------------------------
def assign_mood(row):
    if row["energy"] >= 0.75 and row["tempo"] >= 120:
        return "Energetic"
    elif row["valence"] >= 0.6 and row["energy"] >= 0.5:
        return "Happy"
    elif row["acousticness"] >= 0.6 and row["energy"] <= 0.5:
        return "Calm"
    elif row["valence"] <= 0.4 and row["energy"] <= 0.5:
        return "Sad"
    else:
        return "Calm"


df["mood"] = df.apply(assign_mood, axis=1)


# -----------------------------
# Load Saved Model + Scaler
# -----------------------------
knn = joblib.load("knn_mood_model.pkl")
scaler = joblib.load("scaler.pkl")


# -----------------------------
# Scale dataset features (IMPORTANT)
# -----------------------------
X = df[audio_features]
X_scaled = scaler.transform(X)

song_db = df.copy()
song_db[audio_features] = X_scaled


# -----------------------------
# Recommendation Function
# -----------------------------
def recommend_songs_by_mood(user_mood, song_db, audio_features, top_n=5):
    mood_songs = song_db[song_db["mood"] == user_mood]

    if len(mood_songs) == 0:
        return pd.DataFrame()

    features = mood_songs[audio_features].values

    ref_idx = np.random.randint(len(mood_songs))
    ref_vector = features[ref_idx].reshape(1, -1)

    similarity = cosine_similarity(ref_vector, features)[0]
    top_indices = similarity.argsort()[::-1][1:top_n + 1]

    return mood_songs.iloc[top_indices][
        ["track_name", "artist_name", "mood"]
    ]


# -----------------------------
# User Input (Interactive)
# -----------------------------
print("\n🎧 FeelTheBeat – Emotion Based Music Recommendation")
print("Available moods: Happy, Sad, Calm, Energetic\n")

user_mood = input("Enter your mood: ").strip().capitalize()

recommendations = recommend_songs_by_mood(
    user_mood,
    song_db,
    audio_features,
    top_n=10
)

if recommendations.empty:
    print("\n❌ No songs found for this mood.")
else:
    print("\n✅ Recommended Songs:\n")
    print(recommendations.to_string(index=False))
