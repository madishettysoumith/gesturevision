
import os, json
from typing import List

def ensure_dirs():
    os.makedirs("dataset/static", exist_ok=True)
    os.makedirs("dataset/dynamic", exist_ok=True)
    os.makedirs("models", exist_ok=True)

def save_label_map(classes: List[str], path: str = "models/label_map.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(classes), f, indent=2)

def load_label_map(path: str = "models/label_map.json") -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
