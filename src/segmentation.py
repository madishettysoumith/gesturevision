
import cv2
import numpy as np
import yaml
import os

def load_cfg():
    with open(os.path.join("config", "settings.yaml"), "r") as f:
        return yaml.safe_load(f)

_cfg = load_cfg()

_bg = None
def get_bg_subtractor():
    global _bg
    if _bg is None:
        _bg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)
    return _bg

def skin_mask_hsv(img_bgr):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(_cfg["preprocess"]["skin_hsv_lower"], dtype=np.uint8)
    upper = np.array(_cfg["preprocess"]["skin_hsv_upper"], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.medianBlur(mask, 5)
    return mask

def segment_hand(img_bgr):
    mask_skin = skin_mask_hsv(img_bgr)
    if _cfg["preprocess"]["use_bg_subtractor"]:
        bg = get_bg_subtractor().apply(img_bgr)
        fg = cv2.bitwise_and(mask_skin, bg)
    else:
        fg = mask_skin
    # Morphology to clean
    kernel = np.ones((3,3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Apply mask
    hand = cv2.bitwise_and(img_bgr, img_bgr, mask=fg)
    return hand, fg

def to_model_input(hand_bgr, img_size=128):
    # Convert to gray after segmentation; resize; normalize [0,1]
    gray = cv2.cvtColor(hand_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (img_size, img_size), interpolation=cv2.INTER_AREA)
    gray = gray.astype("float32") / 255.0
    gray = np.expand_dims(gray, axis=-1)  # (H,W,1)
    return gray
