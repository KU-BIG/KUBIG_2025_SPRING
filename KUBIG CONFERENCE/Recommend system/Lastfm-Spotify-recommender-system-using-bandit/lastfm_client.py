import requests
from config import LASTFM_API_KEY

class LastFMClient:
    def __init__(self):
        self.api_key = LASTFM_API_KEY
        self.base_url = "http://ws.audioscrobbler.com/2.0/"

    def _make_request(self, method, params):
        params.update({
            "method": method,
            "api_key": self.api_key,
            "format": "json"
        })
        response = requests.get(self.base_url, params=params)
        return response.json()

    def get_similar_tracks(self, track_name, artist_name, limit=10):
        result = self._make_request("track.getSimilar", {
            "track": track_name,
            "artist": artist_name,
            "limit": limit
        })
        return result.get("similartracks", {}).get("track", [])

    def get_top_tracks_by_artist(self, artist_name, limit=10):
        result = self._make_request("artist.getTopTracks", {
            "artist": artist_name,
            "limit": limit
        })
        return result.get("toptracks", {}).get("track", [])

    def get_top_tracks_by_tag(self, tag, limit=10):
        result = self._make_request("tag.getTopTracks", {
            "tag": tag,
            "limit": limit
        })
        return result.get("tracks", {}).get("track", [])
