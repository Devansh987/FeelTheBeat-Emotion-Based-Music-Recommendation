import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse

# -----------------------------
# 1. Page Config
# -----------------------------
st.set_page_config(
    page_title="Feel The Beat - AI Music Discovery",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# 2. State Initialization
# -----------------------------
if 'selected_vibe' not in st.session_state:
    st.session_state.selected_vibe = "Happy"

# -----------------------------
# 3. Custom CSS Design System
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,600;0,700;0,800;1,400;1,600&family=Caveat:wght@600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        color: #ffffff;
    }

    .stApp {
        background: #090714;
        background-image: 
            radial-gradient(at 0% 0%, rgba(124, 58, 237, 0.15) 0px, transparent 50%),
            radial-gradient(at 100% 0%, rgba(219, 39, 119, 0.15) 0px, transparent 50%),
            radial-gradient(at 50% 100%, rgba(15, 23, 42, 0.8) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* Top Navigation Bar */
    .nav-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 24px;
        background: rgba(15, 12, 29, 0.6);
        backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        margin-bottom: 30px;
    }

    .nav-logo {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.5px;
        background: linear-gradient(90deg, #a78bfa, #f472b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .nav-links {
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .nav-item {
        padding: 8px 18px;
        border-radius: 20px;
        font-size: 14px;
        font-weight: 600;
        color: #94a3b8;
        text-decoration: none;
        transition: all 0.2s ease;
        cursor: pointer;
    }

    .nav-item.active {
        background: rgba(124, 58, 237, 0.25);
        color: #ffffff;
        border: 1px solid rgba(167, 139, 250, 0.3);
    }

    .nav-user {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .user-avatar {
        width: 34px;
        height: 34px;
        border-radius: 50%;
        background: linear-gradient(135deg, #7c3aed, #db2777);
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 14px;
    }

    /* Hero Section */
    .hero-wrapper {
        position: relative;
        text-align: center;
        padding: 40px 20px 30px 20px;
        overflow: hidden;
    }

    .handwriting-left {
        position: absolute;
        left: 5%;
        top: 25%;
        font-family: 'Caveat', cursive;
        font-size: 32px;
        color: #8b5cf6;
        transform: rotate(-12deg);
        opacity: 0.7;
    }

    .handwriting-right {
        position: absolute;
        right: 4%;
        top: 60%;
        font-family: 'Caveat', cursive;
        font-size: 28px;
        color: #ec4899;
        transform: rotate(8deg);
        opacity: 0.7;
    }

    .vinyl-graphic {
        position: absolute;
        right: 8%;
        top: 0%;
        width: 220px;
        height: 220px;
        border-radius: 50%;
        background: radial-gradient(circle, #1e1b4b 20%, #000 70%);
        border: 4px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 50px rgba(124, 58, 237, 0.25);
        display: flex;
        align-items: center;
        justify-content: center;
        animation: spin 20s linear infinite;
        opacity: 0.6;
        z-index: 0;
    }

    .vinyl-inner {
        width: 70px;
        height: 70px;
        border-radius: 50%;
        background: linear-gradient(135deg, #7c3aed, #db2777);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    @keyframes spin {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 750px;
        margin: 0 auto;
    }

    .hero-badge {
        display: inline-block;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #a78bfa;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-size: 64px;
        font-weight: 900;
        letter-spacing: -2px;
        line-height: 1.05;
        margin-bottom: 12px;
        background: linear-gradient(90deg, #ffffff 30%, #c084fc 70%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 22px;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 12px;
    }

    .hero-desc {
        font-size: 15px;
        color: #94a3b8;
        line-height: 1.6;
        margin-bottom: 30px;
    }

    /* Controls Panel Card */
    .control-card {
        background: rgba(18, 14, 38, 0.7);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 28px;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
        margin-bottom: 40px;
    }

    /* Big Action Button */
    div.stButton > button {
        width: 100%;
        height: 56px;
        border-radius: 16px;
        border: none;
        background: linear-gradient(90deg, #6366f1, #a855f7, #ec4899);
        background-size: 200% auto;
        color: white;
        font-size: 18px;
        font-weight: 800;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: all 0.4s ease;
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.4);
    }

    div.stButton > button:hover {
        background-position: right center;
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(236, 72, 153, 0.6);
    }

    /* Popular Vibes Section */
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }

    .section-title {
        font-size: 20px;
        font-weight: 800;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    /* Track Cards */
    .song-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        padding: 20px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
        position: relative;
    }

    .song-card:hover {
        transform: translateY(-4px);
        background: rgba(255, 255, 255, 0.06);
        border-color: rgba(167, 139, 250, 0.4);
        box-shadow: 0 12px 30px rgba(124, 58, 237, 0.2);
    }

    .song-title {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .song-artist {
        font-size: 14px;
        color: #a78bfa;
        font-weight: 600;
        margin-bottom: 14px;
    }

    .spotify-btn {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 14px;
        border-radius: 20px;
        background: #1db954;
        color: #000;
        font-size: 12px;
        font-weight: 700;
        text-decoration: none;
        transition: all 0.2s ease;
    }

    .spotify-btn:hover {
        background: #1ed760;
        transform: scale(1.04);
    }

    .stat-bar-container {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
        margin-top: 14px;
    }

    .stat-item {
        background: rgba(0, 0, 0, 0.3);
        padding: 8px 12px;
        border-radius: 10px;
    }

    .stat-label-val {
        display: flex;
        justify-content: space-between;
        font-size: 11px;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 4px;
        text-transform: uppercase;
    }

    .progress-track {
        height: 6px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        overflow: hidden;
    }

    .progress-bar {
        height: 100%;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# 4. Top Navigation Bar UI
# -----------------------------
st.markdown("""
<div class="nav-container">
    <div class="nav-logo">
        <span>🎧</span> Feel The Beat
    </div>
    <div class="nav-links">
        <span class="nav-item active">🏠 Home</span>
        <span class="nav-item">🧭 Discover</span>
        <span class="nav-item">🎵 My Playlists</span>
        <span class="nav-item">ⓘ About</span>
    </div>
    <div class="nav-user">
        <span style="font-size:18px; cursor:pointer;">☀️</span>
        <div class="user-avatar">D</div>
        <span style="font-size:14px; font-weight:600; color:#cbd5e1;">Devansh ▾</span>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 5. Hero Banner Section UI
# -----------------------------
st.markdown("""
<div class="hero-wrapper">
    <div class="handwriting-left">Music Understands You</div>
    <div class="handwriting-right">Different Moods<br>New Stories</div>
    <div class="vinyl-graphic">
        <div class="vinyl-inner">🎵</div>
    </div>
    <div class="hero-content">
        <div class="hero-badge">AI-POWERED MUSIC DISCOVERY</div>
        <div class="hero-title">FEEL THE BEAT</div>
        <div class="hero-subtitle">Turn your emotions into the perfect playlist</div>
        <div class="hero-desc">
            Discover music that matches your mood using AI and Spotify's audio features.<br>
            Select a vibe, set your preferences, and let the music find you.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# 6. Load Dataset & Scaler
# -----------------------------
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("SpotifyFeatures.csv")
        df.dropna(how="all", inplace=True)
        df.drop_duplicates(subset="track_id", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_resource
def load_scaler_model():
    try:
        scaler = joblib.load("scaler.pkl")
        knn = joblib.load("knn_mood_model.pkl")
        return scaler, knn
    except Exception:
        return None, None

df = load_dataset()
scaler, knn = load_scaler_model()

audio_features = [
    "acousticness", "danceability", "energy",
    "instrumentalness", "liveness", "loudness",
    "speechiness", "tempo", "valence"
]

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
    if scaler is not None:
        X = df[audio_features]
        X_scaled = scaler.transform(X)
        song_db = df.copy()
        song_db[audio_features] = X_scaled
    else:
        song_db = df.copy()
else:
    song_db = pd.DataFrame()

# Vibe mapping to database mood
vibe_mood_map = {
    "Happy": "Happy",
    "Sad": "Sad",
    "Energetic": "Energetic",
    "Romantic": "Sad",
    "Chill": "Calm",
    "Focus": "Calm",
    "Sleep": "Calm",
    "Workout": "Energetic",
    "Party": "Energetic"
}

# -----------------------------
# 7. Central Controls Box UI
# -----------------------------
with st.container():
    st.markdown('<div class="control-card">', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.2, 1, 1.8])
    
    with c1:
        st.markdown("**1. Select Vibe**")
        vibe_list = ["Happy", "Sad", "Energetic", "Romantic", "Chill", "Focus", "Sleep", "Workout", "Party"]
        selected_vibe = st.selectbox(
            "Select Vibe",
            vibe_list,
            index=vibe_list.index(st.session_state.selected_vibe) if st.session_state.selected_vibe in vibe_list else 0,
            label_visibility="collapsed"
        )
        st.session_state.selected_vibe = selected_vibe

    with c2:
        st.markdown("**2. Quantity**")
        top_n = st.slider("Quantity", 5, 20, 11, label_visibility="collapsed")

    with c3:
        st.markdown("**3. Filter (Optional)**")
        artist_filter = st.text_input("Filter", placeholder="e.g. The Weeknd, Arijit Singh, Party...", label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    generate_btn = st.button("✨ GENERATE PLAYLIST →")
    st.markdown("<div style='text-align:center; color:#94a3b8; font-size:13px; margin-top:8px;'>✨ Let AI curate the perfect tracks for your mood</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 8. Popular Vibes Grid Section
# -----------------------------
st.markdown("""
<div class="section-header">
    <div class="section-title">
        <span style="color:#a78bfa;">📊</span> Popular Vibes
    </div>
    <div style="font-size:13px; color:#94a3b8;">
        Choose a vibe to get started ⤴
    </div>
</div>
""", unsafe_allow_html=True)

vibe_data = [
    {"name": "Happy", "icon": "😊", "desc": "Feel good, stay bright"},
    {"name": "Sad", "icon": "🙁", "desc": "It's okay to feel"},
    {"name": "Energetic", "icon": "⚡", "desc": "Fuel your day"},
    {"name": "Romantic", "icon": "💖", "desc": "For the hearts"},
    {"name": "Chill", "icon": "☀️", "desc": "Relax and unwind"},
    {"name": "Focus", "icon": "🧘", "desc": "Deep work vibes"},
    {"name": "Sleep", "icon": "🌙", "desc": "Drift into peace"},
    {"name": "Workout", "icon": "🎮", "desc": "Push your limits"},
    {"name": "Party", "icon": "🎵", "desc": "Turn it up"}
]

v_cols = st.columns(9)
for idx, vibe in enumerate(vibe_data):
    with v_cols[idx]:
        is_selected = (st.session_state.selected_vibe == vibe["name"])
        vibe_label = vibe["icon"] + "\n" + vibe["name"]
        btn_click = st.button(
            vibe_label,
            key=f"vibe_btn_{idx}",
            use_container_width=True
        )
        if btn_click:
            st.session_state.selected_vibe = vibe["name"]
            st.rerun()

st.write("")
st.write("")

# -----------------------------
# 9. Playlist Recommendation Engine & Display
# -----------------------------
def get_recommendations(user_vibe, top_n=10):
    target_mood = vibe_mood_map.get(user_vibe, "Happy")
    mood_songs = song_db[song_db["mood"] == target_mood]
    
    if len(mood_songs) == 0:
        return pd.DataFrame()

    features = mood_songs[audio_features].values
    ref_idx = np.random.randint(len(mood_songs))
    ref_vector = features[ref_idx].reshape(1, -1)

    similarity = cosine_similarity(ref_vector, features)[0]
    top_indices = similarity.argsort()[::-1][1:top_n + 1]

    return mood_songs.iloc[top_indices].copy()

if generate_btn:
    if df.empty:
        st.error("Dataset not found. Please ensure 'SpotifyFeatures.csv' is present.")
    else:
        with st.spinner("AI is analyzing sonic features and crafting your playlist..."):
            recs = get_recommendations(st.session_state.selected_vibe, top_n=top_n)

            if artist_filter.strip():
                recs = recs[
                    recs["artist_name"].str.contains(artist_filter, case=False, na=False) |
                    recs["track_name"].str.contains(artist_filter, case=False, na=False)
                ]

            if recs.empty:
                st.warning(f"No tracks found matching vibe '{st.session_state.selected_vibe}' with filter '{artist_filter}'.")
            else:
                st.markdown(f"""
                <div style="text-align: center; margin: 30px 0 20px 0;">
                    <h2 style="font-size: 28px; font-weight: 800;">🔥 CURATED PLAYLIST FOR <span style="color:#ec4899;">{st.session_state.selected_vibe.upper()}</span></h2>
                    <p style="color:#94a3b8; font-size:14px;">Handpicked {len(recs)} tracks based on acoustic similarity</p>
                </div>
                """, unsafe_allow_html=True)

                col_left, col_right = st.columns(2)

                for i, (index, row) in enumerate(recs.iterrows()):
                    real_row = df[df["track_id"] == row["track_id"]].iloc[0]

                    energy_pct = int(float(real_row["energy"]) * 100)
                    dance_pct = int(float(real_row["danceability"]) * 100)
                    val_pct = int(float(real_row["valence"]) * 100)
                    acous_pct = int(float(real_row["acousticness"]) * 100)

                    search_query = urllib.parse.quote(f"{row['track_name']} {row['artist_name']}")
                    spotify_url = f"https://open.spotify.com/search/{search_query}"

                    card_html = f"""
                    <div class="song-card">
                        <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                            <div style="max-width: 75%;">
                                <div class="song-title">#{i+1} {row['track_name']}</div>
                                <div class="song-artist">🎤 {row['artist_name']}</div>
                            </div>
                            <a href="{spotify_url}" target="_blank" class="spotify-btn">
                                <span>▶</span> Spotify
                            </a>
                        </div>
                        <div class="stat-bar-container">
                            <div class="stat-item">
                                <div class="stat-label-val"><span>Energy</span><span>{energy_pct}%</span></div>
                                <div class="progress-track">
                                    <div class="progress-bar" style="width: {energy_pct}%; background: linear-gradient(90deg, #ec4899, #f43f5e);"></div>
                                </div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label-val"><span>Dance</span><span>{dance_pct}%</span></div>
                                <div class="progress-track">
                                    <div class="progress-bar" style="width: {dance_pct}%; background: linear-gradient(90deg, #3b82f6, #06b6d4);"></div>
                                </div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label-val"><span>Valence</span><span>{val_pct}%</span></div>
                                <div class="progress-track">
                                    <div class="progress-bar" style="width: {val_pct}%; background: linear-gradient(90deg, #10b981, #34d399);"></div>
                                </div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-label-val"><span>Acoustic</span><span>{acous_pct}%</span></div>
                                <div class="progress-track">
                                    <div class="progress-bar" style="width: {acous_pct}%; background: linear-gradient(90deg, #f59e0b, #fbbf24);"></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """

                    if i % 2 == 0:
                        col_left.markdown(card_html, unsafe_allow_html=True)
                    else:
                        col_right.markdown(card_html, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align: center; margin-top: 40px; padding: 40px; background: rgba(255,255,255,0.02); border-radius: 20px; border: 1px dashed rgba(255,255,255,0.1);">
        <h3 style="color:#a78bfa; font-weight:700;">READY TO DISCOVER YOUR BEAT</h3>
        <p style="color:#94a3b8; font-size:14px;">Select your mood above and click 'GENERATE PLAYLIST' to begin.</p>
    </div>
    """, unsafe_allow_html=True)