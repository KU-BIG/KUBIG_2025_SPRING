import requests
import urllib.parse
from config import SPOTIFY_ACCESS_TOKEN

def search_track_on_spotify(track_name, artist_name=None):
    query = f"{track_name} {artist_name}" if artist_name else track_name
    url = f"https://api.spotify.com/v1/search?q={urllib.parse.quote(query)}&type=track&limit=1"
    headers = {
        "Authorization": f"Bearer {SPOTIFY_ACCESS_TOKEN}"
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print("❌ 요청 실패:", response.status_code, response.text)
        return

    data = response.json()
    items = data.get("tracks", {}).get("items", [])
    if not items:
        print("❗ 검색 결과 없음")
        return

    track = items[0]
    print("🎵 제목:", track["name"])
    print("🎤 아티스트:", track["artists"][0]["name"])
    print("🟢 Spotify URI:", track["uri"])

# 테스트 실행
search_track_on_spotify("Gravity", "John Mayer")