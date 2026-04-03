import cv2
import numpy as np

def show(title, img, scale=1.0):
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(title, img_resized)
    cv2.waitKey(0)

def show_cells_grid(cells):
    rows = []
    for i in range(9):
        row = np.hstack(cells[i*9:(i+1)*9])
        rows.append(row)
    grid = np.vstack(rows)
    cv2.imshow("extracted cells", grid)
    cv2.waitKey(0)
