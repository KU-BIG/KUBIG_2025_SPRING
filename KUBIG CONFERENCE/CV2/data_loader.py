from PIL import Image
import requests

def load_image_from_path(path: str) -> Image.Image:
    return Image.open(path).convert("RGB")

def load_image_from_url(url: str) -> Image.Image:
    return Image.open(requests.get(url, stream=True).raw).convert("RGB")
