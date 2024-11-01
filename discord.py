import requests
from os import remove
from utils import get_config

webhook_url = get_config("WEBHOOK")

def image_to_discord(image_path, model):
    if image_path.endswith(".png"):
        with open(image_path, 'rb') as f:
            files = {
                'file': (image_path, f)
            }
            requests.post(webhook_url, files=files, data={"content": model})

        remove(image_path)