from src.capture import capture_grid, debug_capture_grid
from src.cell_extraction import extract_cells_from_grid, debug_extract_cells_from_grid, save_sudoku_dataset, extract_cells_from_grid_inv
from src.solver import solve_sudoku
from src.interact import get_cells_to_fill, fill_on_sudoku_com, quad_to_region
from src.tesseract_module import recognize_with_tesseract
from src.train_cnn import train_MNIST, train_from_folder
from src.DigitRecognizer import DigitRecognizer
from src.TemplateMatcher import TemplateMatcher
import numpy as np
import time

def main_pipeline():
    print("Choose method :")
    print("1 - Tesseract")
    print("2 - Template Matching")
    print("3 - CNN")
    print("4 - Train cnn model")

    choice = input("> ")

    time.sleep(1)

    quad, grid = capture_grid()
    digits = []
    cells = []

    if choice == "1":
        cells = extract_cells_from_grid_inv(grid)
        digits = recognize_with_tesseract(cells)

    elif choice == "2":
        cells = extract_cells_from_grid(grid)
        matcher = TemplateMatcher()
        digits = [matcher.predict(c) for c in cells]

    elif choice == "3":
        cells = extract_cells_from_grid_inv(grid)
        recognizer = DigitRecognizer()
        digits = [recognizer.predict(c) for c in cells]

    elif choice == "4":
        print("Choose TYPE :")
        print("1 - Sudoku grid")
        print("2 - MNIST")
        type = input("> ")

        if type == "1":
            cells = extract_cells_from_grid_inv(grid)
            save_sudoku_dataset(cells)
            train_from_folder()
        elif type == "2":
            train_MNIST()
        return 0

    else:
        print("Invalid option")
        return 1

    sudoku_grid = np.array(digits).reshape((9, 9))
    print(sudoku_grid)

    og_grid = sudoku_grid.copy()

    solve_sudoku(sudoku_grid)

    if np.array_equal(og_grid, sudoku_grid):
        print("Cannot solve Sudoku")
        return 1

    print("Fill cells on Sudoku ?")
    print("1 - Yes")
    print("2 - No")

    fill = input("> ")
    if fill == "1":
        cells_to_fill = get_cells_to_fill(og_grid, sudoku_grid)
        fill_on_sudoku_com(cells_to_fill, quad_to_region(quad))

    

if __name__ == "__main__":
    main_pipeline()