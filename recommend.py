import os, random, time
from collections import deque

# direktori gambar
ASSETS_DIR = os.path.join("assets", "images")

# mapping emosi -> kategori rasa
EMOTION_TO_TASTE = {
    "happy": "manis",
    "sad": "pahit",
    "neutral": "asam",
}

# daftar menu (nama -> file .jpg)
COFFEE_MENU = {
    "manis": [
        ("Caramel Macchiato", os.path.join(ASSETS_DIR, "manis", "caramel_macchiato.jpg")),
        ("Vanilla Latte", os.path.join(ASSETS_DIR, "manis", "vanilla_latte.jpg")),
        ("Hazelnut Mocha", os.path.join(ASSETS_DIR, "manis", "hazelnut_mocha.jpg")),
    ],
    "pahit": [
        ("Espresso", os.path.join(ASSETS_DIR, "pahit", "espresso.jpg")),
        ("Long Black", os.path.join(ASSETS_DIR, "pahit", "long_black.jpg")),
        ("Ristretto", os.path.join(ASSETS_DIR, "pahit", "ristretto.jpg")),
    ],
    "asam": [
        ("Americano", os.path.join(ASSETS_DIR, "asam", "americano.jpg")),
        ("Espresso Tonic", os.path.join(ASSETS_DIR, "asam", "espresso_tonic.jpg")),
        ("Kopi Lime", os.path.join(ASSETS_DIR, "asam", "kopi_lime.jpg")),
    ],
}

def get_image_recommendations(emotion, k=2):
    taste = EMOTION_TO_TASTE.get(emotion, None)
    if taste is None or taste not in COFFEE_MENU:
        return []
    menu = COFFEE_MENU[taste]
    if len(menu) <= k:
        return menu
    return random.sample(menu, k)

class RecoMemory:
    def __init__(self, maxlen=10, cooldown_s=10.0):
        self.history = deque(maxlen=maxlen)
        self.cooldown_s = cooldown_s
        self.last_time = {}

    def was_recent(self, label):
        now = time.time()
        if label in self.last_time and (now - self.last_time[label]) < self.cooldown_s:
            return True
        return False

    def push(self, label, items):
        now = time.time()
        self.last_time[label] = now
        self.history.append((label, items, now))