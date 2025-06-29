const client_id = 'd28df89507ca47bebaa9385ebb546e92';  // Replace this
const redirect_uri = 'http://localhost:8000/';
const scopes = 'streaming user-read-email user-read-private user-modify-playback-state user-read-playback-state';

function getTokenFromUrl() {
  return new URLSearchParams(window.location.hash.substring(1)).get('access_token');
}

function redirectToSpotifyLogin() {
  const authUrl = `https://accounts.spotify.com/authorize?client_id=${client_id}&response_type=token&redirect_uri=${encodeURIComponent(redirect_uri)}&scope=${encodeURIComponent(scopes)}`;
  window.location = authUrl;
}

document.getElementById('login-btn').onclick = redirectToSpotifyLogin;

window.onSpotifyWebPlaybackSDKReady = () => {
  const token = getTokenFromUrl();
  if (!token) return;

  document.getElementById('login-area').style.display = 'none';
  document.getElementById('player-area').style.display = 'block';

  fetch('https://api.spotify.com/v1/me', {
    headers: { Authorization: 'Bearer ' + token }
  })
    .then(res => res.json())
    .then(data => {
      document.getElementById('display-name').innerText = data.display_name;
    });

  const player = new Spotify.Player({
    name: 'Web Playback SDK',
    getOAuthToken: cb => cb(token),
    volume: 0.8
  });

  let device_id = null;

  player.addListener('ready', ({ device_id: id }) => {
    device_id = id;
    console.log('Ready with Device ID', device_id);
  });

  player.connect();

  document.getElementById('play-btn').onclick = () => {
    const uriString = new URLSearchParams(window.location.search).get('uris');
    const uris = uriString ? uriString.split(',') : [];
    if (!device_id || uris.length === 0) {
      alert("Missing device or track list");
      return;
    }
    fetch(`https://api.spotify.com/v1/me/player/play?device_id=${device_id}`, {
      method: 'PUT',
      body: JSON.stringify({ uris: uris }),
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`
      }
    });
  };
};
