import pyautogui
import time

def get_cells_to_fill(original, solved):
    to_fill = []
    for i in range(9):
        for j in range(9):
            if original[i][j] == 0:
                to_fill.append((i, j, solved[i][j]))
    return to_fill

def quad_to_region(quad):
    xs = [p[0][0] for p in quad]
    ys = [p[0][1] for p in quad]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    w = x_max - x_min
    h = y_max - y_min

    return (x_min, y_min, w, h)

def fill_on_sudoku_com(cells_to_fill, region):
    x0, y0, w, h = region
    cell_w = w // 9
    cell_h = h // 9

    for (i, j, value) in cells_to_fill:
        x = x0 + j * cell_w + cell_w // 2
        y = y0 + i * cell_h + cell_h // 2

        pyautogui.click(x, y)
        pyautogui.press(str(value))
        time.sleep(0.25)
