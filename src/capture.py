import cv2
import numpy as np
import pyautogui
from src.grid_detection import preprocess_for_grid, find_largest_quad, resize_grid, manual_grid_select
from src.utils import show

def capture_grid():
    screenshot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    tresh = preprocess_for_grid(img=img)

    quad = find_largest_quad(tresh)
    if quad is None:
        raise Exception("Can't find grid")

    resized = resize_grid(img=img, quad=quad)
    return quad, resized

def debug_capture_grid():
    # region = manual_grid_select()
    # screenshot = pyautogui.screenshot(region=region)

    screenshot = pyautogui.screenshot()
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    show("Base img", img, scale=0.7)

    tresh = preprocess_for_grid(img=img)
    show("Preprocess", img, scale=0.7)

    quad = find_largest_quad(tresh)
    img_contours = img.copy()
    if quad is None:
        raise Exception("Can't find grid")
    cv2.drawContours(img_contours, [quad], -1, (0, 255, 0), 3)
    show("Contour", img_contours, scale=0.7)

    resized = resize_grid(img=img, quad=quad)
    show("Resized grid", resized, scale=0.7)
    return resized
