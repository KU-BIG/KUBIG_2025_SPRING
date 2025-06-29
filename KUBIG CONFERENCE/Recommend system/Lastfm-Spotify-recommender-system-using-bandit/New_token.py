import os
import base64
import requests

client_id = os.getenv("SPOTIFY_CLIENT_ID")
client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")

auth_str = f"{client_id}:{client_secret}"
b64_auth = base64.b64encode(auth_str.encode()).decode()

headers = {
    "Authorization": f"Basic {b64_auth}",
    "Content-Type": "application/x-www-form-urlencoded"
}
data = {"grant_type": "client_credentials"}

res = requests.post("https://accounts.spotify.com/api/token", headers=headers, data=data)
print("✅ New access token:", res.json().get("access_token"))