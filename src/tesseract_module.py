import cv2
import numpy as np
import pytesseract
from src.utils import show, show_cells_grid

def prepare_for_tesseract(cell):
    # Vérifier que l'image existe
    if cell is None:
        return None

    # Verifier si ndarray
    if not isinstance(cell, np.ndarray):
        return None

    cell = remove_cell_border(cell)

    # Convertir en niveaux de gris si nécessaire
    if cell.ndim == 3:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)

    # Inverser (Tesseract préfère noir sur blanc)
    inv = cv2.bitwise_not(cell)

    # Aggrandir
    resized = cv2.resize(inv, (200, 200), interpolation=cv2.INTER_LINEAR)

    # Épaissir le chiffre
    kernel = np.ones((2, 2), np.uint8)
    thick = cv2.dilate(resized, kernel, iterations=1)

    return thick

def tesseract_digit(cell):
    prep = prepare_for_tesseract(cell)

    if prep is None:
        return 0  # cellule vide

    config = "--psm 13 -c tessedit_char_whitelist=123456789"

    text = pytesseract.image_to_string(prep, config=config)
    text = text.strip()

    print(text)
    if text.isdigit():
        return int(text)

    return 0

def recognize_with_tesseract(cells):
    digits = []

    for cell in cells:
        debug_tesseract(cell)
        digit = tesseract_digit(cell)
        digits.append(digit)

    return np.array(digits).reshape((9, 9))

def debug_tesseract(cell):
    prep = prepare_for_tesseract(cell)
    if prep is not None:
        cv2.imshow("Tesseract input", prep)
        cv2.waitKey(0)

def remove_cell_border(cell):
    h, w = cell.shape[:2]

    # enlever 15% du bord
    margin_h = int(h * 0.15)
    margin_w = int(w * 0.15)

    cropped = cell[margin_h:h-margin_h, margin_w:w-margin_w]
    return cropped