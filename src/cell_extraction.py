from src.capture import capture_grid
from src.utils import show, show_cells_grid

import cv2
import numpy as np

def extract_cells(grid):
    h, w = grid.shape[:2]
    cell_h = h // 9
    cell_w = w // 9

    cells = []

    for i in range(9):
        for j in range(9):
            y1 = i * cell_h
            y2 = (i + 1) * cell_h
            x1 = j * cell_w
            x2 = (j + 1) * cell_w

            cell = grid[y1:y2, x1:x2]
            cells.append(cell)

    return cells

def clean_cell(cell):
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    return thresh

def preprocess_cells(cells):
    return [clean_cell(c) for c in cells]


def extract_cells_from_grid(grid):

    show("Grid", grid, scale=0.7)

    cells = extract_cells(grid)
    clean_cells = preprocess_cells(cells)

    show_cells_grid(clean_cells)

    return clean_cells
