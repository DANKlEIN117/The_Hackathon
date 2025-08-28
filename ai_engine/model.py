import os
import requests
from dotenv import load_dotenv

# Load your token
load_dotenv()
HF_API_KEY = os.getenv("HF_TOKEN")

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased"
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"⚠️ API Error: {response.status_code} - {response.text}"
    return response.json()

print(query({"inputs": "Hello world, I feel great today!"}))
