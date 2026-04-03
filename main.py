from src.capture import capture_grid, debug_capture_grid
from src.cell_extraction import extract_cells_from_grid
from src.binding import bind_cells
from src.solver import solve_sudoku
from src.interact import interact_website
import cv2

def main():
    grid = capture_grid()
    extract_cells_from_grid(grid)

main()