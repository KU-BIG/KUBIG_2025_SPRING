import requests
import urllib.parse
from spotify_auth import get_spotify_access_token

def search_track_on_spotify(track_name, artist_name=None, access_token=None):
    if access_token is None:
        access_token = get_spotify_access_token()
    query = f"{track_name} {artist_name}" if artist_name else track_name
    url = f"https://api.spotify.com/v1/search?q={urllib.parse.quote(query)}&type=track&limit=3"
    headers = {"Authorization": f"Bearer {access_token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        items = response.json()["tracks"]["items"]
        if items:
            uri = items[0]["uri"]
            return {"uri": uri, "id": uri.split(":")[-1]}
    return {"uri": "", "id": ""}
