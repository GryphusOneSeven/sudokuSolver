from src.capture import capture_grid, debug_capture_grid, get_quad
from src.cell_extraction import extract_cells_from_grid
from src.binding import bind_cells
from src.solver import solve_sudoku
from src.interact import get_cells_to_fill, fill_on_sudoku_com, quad_to_region
from src.tesseract_module import recognize_with_tesseract

def main():
    quad, grid = capture_grid()
    cells = extract_cells_from_grid(grid)
    sudoku_grid = recognize_with_tesseract(cells)
    og_grid = sudoku_grid.copy()
    solve_sudoku(sudoku_grid)
    cells_to_fill = get_cells_to_fill(og_grid, sudoku_grid)
    fill_on_sudoku_com(cells_to_fill, quad_to_region(quad))

main()