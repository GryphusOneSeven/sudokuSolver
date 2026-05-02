def find_empty(grid):
    for i in range(9):
        for j in range(9):
            if grid[i][j] == 0:
                return i, j
    return None

def is_valid(grid, num, pos):
    row, col = pos

    # ligne
    if num in grid[row]:
        return False

    # colonne
    if num in grid[:, col]:
        return False

    # carré 3x3
    box_x = col // 3
    box_y = row // 3

    for i in range(box_y*3, box_y*3 + 3):
        for j in range(box_x*3, box_x*3 + 3):
            if grid[i][j] == num:
                return False

    return True

def solve_sudoku(grid):
    empty = find_empty(grid)
    if not empty:
        return True

    row, col = empty

    for num in range(1, 10):
        if is_valid(grid, num, (row, col)):
            grid[row][col] = num

            if solve_sudoku(grid):
                return True

            grid[row][col] = 0

    return False
