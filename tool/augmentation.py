import numpy as np
from PIL import Image
import random

# Aynı bozulma parametreleri (Phase 5'teki ile aynı)
DEGRADATION_PARAMS = {
    "darkness": {
        1: {"gamma": 2.0},
        2: {"gamma": 3.5},
        3: {"gamma": 5.0},
    },
    "noise": {
        1: {"sigma": 15},
        2: {"sigma": 35},
        3: {"sigma": 60},
    },
    "rain": {
        1: {"num_streaks": 300,  "brightness_factor": 0.85, "streak_length": (10, 20), "streak_alpha": 0.3},
        2: {"num_streaks": 700,  "brightness_factor": 0.65, "streak_length": (15, 35), "streak_alpha": 0.5},
        3: {"num_streaks": 1200, "brightness_factor": 0.45, "streak_length": (20, 50), "streak_alpha": 0.7},
    },
}

def apply_darkness(img_array, gamma):
    normalized = img_array.astype(np.float32) / 255.0
    darkened = np.power(normalized, gamma)
    return np.clip(darkened * 255, 0, 255).astype(np.uint8)

def apply_noise(img_array, sigma, rng):
    noise = rng.normal(0, sigma, img_array.shape).astype(np.float32)
    noisy = img_array.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def apply_rain(img_array, num_streaks, brightness_factor, streak_length, streak_alpha, rng):
    h, w, c = img_array.shape
    result = (img_array.astype(np.float32) * brightness_factor).astype(np.float32)
    rain_layer = np.zeros((h, w), dtype=np.float32)
    min_len, max_len = streak_length

    for _ in range(num_streaks):
        x = rng.integers(0, w)
        y = rng.integers(0, h)
        length = rng.integers(min_len, max_len + 1)
        dx = rng.integers(-2, 3)
        thickness = rng.integers(1, 3)

        for t in range(length):
            py = y + t
            px = x + int(dx * t / max(length, 1))
            if 0 <= py < h and 0 <= px < w:
                for offset in range(thickness):
                    if 0 <= px + offset < w:
                        rain_layer[py, px + offset] = 200 + rng.integers(0, 56)

    rain_3ch = np.stack([rain_layer] * 3, axis=-1)
    result = result * (1 - streak_alpha * (rain_layer > 0).astype(np.float32)[..., None]) \
             + rain_3ch * streak_alpha
    return np.clip(result, 0, 255).astype(np.uint8)


class ChallengingConditionTransform:
    """
    Eğitim sırasında resimlere rastgele zorlu koşul ekleyen PyTorch transformu.
    Phase 6 fine-tuning aşaması için veri artırımı sağlar.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        img_array = np.array(img)
        condition = random.choice(["darkness", "noise", "rain"])
        level = random.choice([1, 2, 3])
        params = DEGRADATION_PARAMS[condition][level]
        
        rng = np.random.default_rng()

        if condition == "darkness":
            result = apply_darkness(img_array, params["gamma"])
        elif condition == "noise":
            result = apply_noise(img_array, params["sigma"], rng)
        elif condition == "rain":
            result = apply_rain(
                img_array,
                params["num_streaks"],
                params["brightness_factor"],
                params["streak_length"],
                params["streak_alpha"],
                rng
            )

        return Image.fromarray(result)
