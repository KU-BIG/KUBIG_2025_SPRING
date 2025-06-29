import streamlit as st
from bandit.thompson_sampling import ThompsonSampling
from recommender import Recommender
from spotify_player import search_track_on_spotify
import urllib.parse

st.set_page_config(page_title="🎵 추천 시스템", layout="wide")
st.title("🎧 Last.fm + Spotify Bandit Algorithm 기반 음악 추천 시스템")

# 알고리즘 초기화
if st.sidebar.button("🔄 알고리즘 초기화"):
    st.session_state.recommender = Recommender(ThompsonSampling())
    st.session_state.tracks = []
    st.session_state.feedback = {}
    st.sidebar.success("알고리즘이 초기화되었습니다.")

if "recommender" not in st.session_state:
    st.session_state.recommender = Recommender(ThompsonSampling())
    st.session_state.tracks = []
    st.session_state.feedback = {}

option = st.selectbox("추천 기준을 선택하세요", ["최근 들은 곡", "좋아하는 아티스트", "좋아하는 장르", "나만의 추천"])
track_name = artist_name = tag = ""

# ────────────────────────────────────────
# 1. 나만의 추천 (5곡 + 0~5점 피드백)
# ────────────────────────────────────────
if option == "나만의 추천":
    if st.button("🎯 개인화 추천 받기"):
        st.session_state.feedback = {}
        tracks = st.session_state.recommender.recommend_personal_top(top_k=5)
        st.session_state.tracks = []

        for t in tracks:
            spotify_info = search_track_on_spotify(t["name"], t["artist"]["name"])
            t["spotify_id"] = spotify_info["id"]
            st.session_state.tracks.append(t)

    if st.session_state.tracks:
        st.subheader("🎯 나만의 추천 곡 리스트")
        for i, track in enumerate(st.session_state.tracks):
            st.markdown(f"### 🎵 {track['name']} - {track['artist']['name']}")
            if track.get("spotify_id"):
                st.components.v1.iframe(
                    f"https://open.spotify.com/embed/track/{track['spotify_id']}",
                    height=80
                )
            score = st.radio(
                f"피드백 ({track['id']})",
                ["보류", "0", "1", "2", "3", "4", "5"],
                index=0,
                horizontal=True,
                key=f"personal_feedback_{i}"
            )
            if score != "보류":
                st.session_state.feedback[track["id"]] = int(score) / 5.0

        if st.button("📥 피드백 반영"):
            for track in st.session_state.tracks:
                if track["id"] in st.session_state.feedback:
                    reward = st.session_state.feedback[track["id"]]
                    st.session_state.recommender.give_feedback(track, reward)
            st.success("피드백이 반영되었습니다 🎉")
            st.session_state.tracks = []

# ────────────────────────────────────────
# 2. 일반 추천 (최근곡, 아티스트, 장르)
# ────────────────────────────────────────
else:
    if option == "최근 들은 곡":
        track_name = st.text_input("트랙 이름", "Not Like Us")
        artist_name = st.text_input("아티스트 이름", "Kendrick Lamar")
    elif option == "좋아하는 아티스트":
        artist_name = st.text_input("아티스트 이름", "John Mayer")
    elif option == "좋아하는 장르":
        tag = st.text_input("장르", "hip-hop")

    if st.button("트랙 추천"):
        st.session_state.feedback = {}
        mode = {"최근 들은 곡": "track", "좋아하는 아티스트": "artist", "좋아하는 장르": "tag"}[option]
        tracks = st.session_state.recommender.recommend_bulk(mode, track_name, artist_name, tag)[:5]
        st.session_state.tracks = []

        for t in tracks:
            spotify_info = search_track_on_spotify(t["name"], t["artist"]["name"])
            t["spotify_id"] = spotify_info["id"]
            st.session_state.tracks.append(t)

    if st.session_state.tracks:
        st.subheader("🔽 추천된 트랙 리스트")
        for i, track in enumerate(st.session_state.tracks):
            st.markdown(f"### 🎵 {track['name']} - {track['artist']['name']}")
            if track.get("spotify_id"):
                st.components.v1.iframe(
                    f"https://open.spotify.com/embed/track/{track['spotify_id']}",
                    height=80
                )
            score = st.radio(
                f"피드백 ({track['id']})",
                ["보류", "0", "1", "2", "3", "4", "5"],
                index=0,
                horizontal=True,
                key=f"feedback_{i}"
            )
            if score != "보류":
                st.session_state.feedback[track["id"]] = int(score) / 5.0

        if st.button("📥 업데이트"):
            for track in st.session_state.tracks:
                if track["id"] in st.session_state.feedback:
                    reward = st.session_state.feedback[track["id"]]
                    st.session_state.recommender.give_feedback(track, reward)
            st.success("추천 알고리즘이 업데이트되었습니다.")

            # 웹 플레이어 URL 생성
            uri_list = [f"spotify:track:{track['spotify_id']}" for track in st.session_state.tracks if track.get("spotify_id")]
            if uri_list:
                uri_query = ",".join(uri_list)
                import os
                if "localhost" in os.environ.get("STREAMLIT_SERVER", "localhost"):
                    url = f"http://localhost:8000/index.html?uris={urllib.parse.quote(uri_query)}"
                else:
                    url = f"https://https://lastfm-spotify-recommender-system-using-bandit-cqxi3ylfbhnnwm5.streamlit.app/index.html?uris={urllib.parse.quote(uri_query)}"
                st.markdown(f"[▶ 웹 플레이어에서 전체 재생하기]({url})", unsafe_allow_html=True)