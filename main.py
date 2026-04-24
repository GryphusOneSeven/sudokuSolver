from src.capture import capture_grid, debug_capture_grid
from src.cell_extraction import extract_cells_from_grid, debug_extract_cells_from_grid, save_sudoku_dataset
from src.binding import bind_cells
from src.solver import solve_sudoku
from src.interact import get_cells_to_fill, fill_on_sudoku_com, quad_to_region
from src.tesseract_module import recognize_with_tesseract
from src.train_cnn import train_and_save
from src.DigitRecognizer import DigitRecognizer
from src.TemplateMatcher import TemplateMatcher
import numpy as np
import time

def main_pipeline():
    time.sleep(1)
    
    quad, grid = capture_grid()
    cells = extract_cells_from_grid(grid)
    # cells = extract_cells_from_no_preprocess(grid)
    # save_sudoku_dataset(cells)

    matcher = TemplateMatcher()
    digits = [matcher.predict(c) for c in cells]

    # sudoku_grid = recognize_with_tesseract(cells)

    
    # recognizer = DigitRecognizer()
    # digits = [recognizer.predict(c) for c in cells]

    # sudoku_grid = np.array(digits).reshape((9, 9))
    # og_grid = sudoku_grid.copy()
    # print(og_grid)
    # solve_sudoku(sudoku_grid)
    # print(sudoku_grid)

    # if np.array_equal(og_grid, sudoku_grid):
    #     print("cant solve sudoku")
    #     return 1

    # cells_to_fill = get_cells_to_fill(og_grid, sudoku_grid)
    # fill_on_sudoku_com(cells_to_fill, quad_to_region(quad))
    

if __name__ == "__main__":
    main_pipeline()