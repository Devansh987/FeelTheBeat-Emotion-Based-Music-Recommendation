import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Page Config
# -----------------------------
st.set_page_config(
    page_title="FeelTheBeat Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# 2. Extreme UI Overhaul (CSS)
# -----------------------------
st.markdown("""
<style>
    /* IMPORT FONTS */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700;900&display=swap');

    /* GLOBAL THEME */
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    /* Background Animation */
    .stApp {
        background: linear-gradient(-45deg, #0f0c29, #302b63, #24243e, #141414);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
    }

    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* HERO SECTION */
    .hero-container {
        text-align: center;
        padding: 60px 20px;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 40px;
        box-shadow: 0 0 50px rgba(124, 58, 237, 0.2);
    }

    .hero-title {
        font-size: 80px;
        font-weight: 900;
        background: linear-gradient(to right, #00c6ff, #0072ff, #f0f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
        line-height: 1.1;
        text-transform: uppercase;
        letter-spacing: -2px;
    }

    .hero-subtitle {
        font-size: 20px;
        color: #ddd;
        font-weight: 300;
        letter-spacing: 1px;
    }

    /* BUTTONS STYLING */
    div.stButton > button {
        width: 100%;
        border-radius: 12px;
        height: 55px;
        font-size: 18px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        background: linear-gradient(90deg, #7c3aed, #db2777);
        border: none;
        color: white;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(219, 39, 119, 0.4);
    }

    div.stButton > button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 8px 25px rgba(219, 39, 119, 0.6);
    }

    /* SONG CARD DESIGN */
    .song-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        animation: fadeInUp 0.5s ease-out forwards;
    }

    .song-card:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    .card-flex {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .song-info h3 {
        margin: 0;
        font-size: 20px;
        color: white;
        font-weight: 700;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 280px;
    }

    .song-info p {
        margin: 5px 0 0 0;
        color: #aaa;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* STAT BARS */
    .stat-container {
        margin-top: 15px;
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }

    .stat-row {
        background: rgba(0,0,0,0.3);
        padding: 8px 12px;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .stat-label {
        font-size: 10px;
        color: #bbb;
        text-transform: uppercase;
    }

    .progress-bg {
        width: 60px;
        height: 6px;
        background: rgba(255,255,255,0.1);
        border-radius: 10px;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        border-radius: 10px;
    }

    /* ANIMATION KEYFRAMES */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

</style>
""", unsafe_allow_html=True)


# -----------------------------
# 3. Load Data & Models
# -----------------------------
@st.cache_data
def load_data():
    try:
        # Ensure you have SpotifyFeatures.csv in your folder
        df = pd.read_csv("SpotifyFeatures.csv")
        df.dropna(how="all", inplace=True)
        df.drop_duplicates(subset="track_id", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        return pd.DataFrame()


@st.cache_resource
def load_model():
    try:
        # Ensure you have these .pkl files or handle the error
        knn = joblib.load("knn_mood_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return knn, scaler
    except:
        return None, None


df = load_data()
knn, scaler = load_model()

audio_features = [
    "acousticness", "danceability", "energy",
    "instrumentalness", "liveness", "loudness",
    "speechiness", "tempo", "valence"
]


# -----------------------------
# 4. Logic Functions
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


if not df.empty:
    df["mood"] = df.apply(assign_mood, axis=1)

    # Scale dataset for recommendations
    if scaler:
        X = df[audio_features]
        X_scaled = scaler.transform(X)
        song_db = df.copy()
        song_db[audio_features] = X_scaled
    else:
        song_db = df.copy()


def recommend_songs_by_mood(user_mood, top_n=10):
    mood_songs = song_db[song_db["mood"] == user_mood]
    if len(mood_songs) == 0:
        return pd.DataFrame()

    features = mood_songs[audio_features].values
    ref_idx = np.random.randint(len(mood_songs))
    ref_vector = features[ref_idx].reshape(1, -1)

    similarity = cosine_similarity(ref_vector, features)[0]
    top_indices = similarity.argsort()[::-1][1:top_n + 1]

    return mood_songs.iloc[top_indices].copy()


# -----------------------------
# 5. UI Layout
# -----------------------------

# --- Header ---
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Feel The Beat</div>
    <div class="hero-subtitle">AI-POWERED SONIC DISCOVERY ENGINE</div>
</div>
""", unsafe_allow_html=True)

# --- Controls ---
with st.container():
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        st.markdown("**1. Select Vibe**")
        selected_mood = st.selectbox(
            "Mood",
            ["Happy", "Sad", "Calm", "Energetic"],
            label_visibility="collapsed"
        )

    with c2:
        st.markdown("**2. Quantity**")
        top_n = st.slider("Count", 5, 20, 8, label_visibility="collapsed")

    with c3:
        st.markdown("**3. Filter (Optional)**")
        artist_filter = st.text_input("Search Artist", placeholder="e.g. The Weeknd...", label_visibility="collapsed")

st.write("")
st.write("")

# --- The Big Trigger Button ---
generate_btn = st.button("🚀 GENERATE PLAYLIST")

# --- Results Section ---
if generate_btn:
    if df.empty:
        st.error("Data not loaded. Please check 'SpotifyFeatures.csv'.")
    else:
        recs = recommend_songs_by_mood(selected_mood, top_n=top_n)

        if artist_filter.strip():
            recs = recs[recs["artist_name"].str.contains(artist_filter, case=False, na=False)]

        if recs.empty:
            st.warning(f"No tracks found for mood '{selected_mood}' with that filter.")
        else:
            st.markdown(
                f"<h3 style='text-align:center; margin-bottom: 30px; color: white;'>🔥 CURATED FOR: <span style='color:#00c6ff'>{selected_mood.upper()}</span></h3>",
                unsafe_allow_html=True)

            col_a, col_b = st.columns(2)

            for i, (index, row) in enumerate(recs.iterrows()):
                # Get raw values for display
                real_row = df[df["track_id"] == row["track_id"]].iloc[0]

                # Format metrics for the progress bars (0 to 100%)
                energy_pct = int(float(real_row["energy"]) * 100)
                dance_pct = int(float(real_row["danceability"]) * 100)
                val_pct = int(float(real_row["valence"]) * 100)
                acous_pct = int(float(real_row["acousticness"]) * 100)

                # Colors based on intensity
                e_color = "#ff0080" if energy_pct > 50 else "#bd00ff"
                d_color = "#00c6ff"

                # ------------------------------------------------------------------
                # IMPORTANT: THE HTML BELOW IS LEFT-ALIGNED TO FIX THE DISPLAY BUG
                # ------------------------------------------------------------------
                card_html = f"""
<div class="song-card" style="animation-delay: {i * 0.1}s">
<div class="card-flex">
<div class="song-info">
<h3>{row['track_name']}</h3>
<p>🎤 {row['artist_name']}</p>
</div>
<div style="font-size: 24px;">💿</div>
</div>
<div class="stat-container">
<div class="stat-row">
<span class="stat-label">Energy</span>
<div class="progress-bg">
<div class="progress-fill" style="width: {energy_pct}%; background: {e_color};"></div>
</div>
</div>
<div class="stat-row">
<span class="stat-label">Dance</span>
<div class="progress-bg">
<div class="progress-fill" style="width: {dance_pct}%; background: {d_color};"></div>
</div>
</div>
<div class="stat-row">
<span class="stat-label">Valence</span>
<div class="progress-bg">
<div class="progress-fill" style="width: {val_pct}%; background: #00ff9d;"></div>
</div>
</div>
<div class="stat-row">
<span class="stat-label">Acoustic</span>
<div class="progress-bg">
<div class="progress-fill" style="width: {acous_pct}%; background: #ffea00;"></div>
</div>
</div>
</div>
</div>
"""

                if i % 2 == 0:
                    col_a.markdown(card_html, unsafe_allow_html=True)
                else:
                    col_b.markdown(card_html, unsafe_allow_html=True)

else:
    # --- Landing State Placeholder ---
    st.markdown("""
    <div style="text-align: center; margin-top: 50px; opacity: 0.5;">
        <h2 style="color:white;">WAITING FOR INPUT...</h2>
        <p style="color:white;">Select a mood above to initialize the recommendation engine.</p>
    </div>
    """, unsafe_allow_html=True)