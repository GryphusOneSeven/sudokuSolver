import pyautogui
import numpy as np
import cv2
import pyautogui

def manual_grid_select():
    print("Top left corner")
    input()
    x1, y1 = pyautogui.position()
    print(f"Point 1 : {x1}, {y1}")

    print("Bottom right corner")
    input()
    x2, y2 = pyautogui.position()
    print(f"Point 2 : {x2}, {y2}")

    area = (x1, y1, x2 - x1, y2 - y1)
    print(f"area : {area}")
    return area

def preprocess_for_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )
    return thresh

def find_largest_quad(thresh):
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    max_area = 0
    best_quad = None

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 10000:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        if len(approx) == 4 and area > max_area:
            best_quad = approx
            max_area = area

    return best_quad

def order_points(pts):
    pts = pts.reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right

    diff = np.diff(pts, axis=1)
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left

    return ordered


def resize_grid(img, quad):
    quad = order_points(quad)
    (tl, tr, br, bl) = quad

    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(quad, dst)
    resized = cv2.warpPerspective(img, M, (maxWidth, maxHeight))

    return resized