# sudokuSolver – Computer Vision

This project implements an automatic Sudoku solver using computer vision and artificial intelligence.  
It detects the Sudoku grid on sudoku.com, extracts the 81 cells, recognizes digits using three different methods, solves the puzzle, and fills the solution automatically.

## 🚀 Features

- Automatic grid detection (OpenCV)
- Cell extraction and preprocessing
- Three digit-recognition methods:
  1. **Tesseract OCR**
  2. **Convolutional Neural Network (CNN)**
  3. **Template Matching (OpenCV)**
- Sudoku solver using backtracking
- Automatic grid filling (PyAutoGUI)
- Docker support for reproducible environments

---

## 1. Clone the repository

```bash
git clone <REPOSITORY_URL>
cd sudoku-solver
```

## 2. Install Python dependencies (outside Docker)

```bash
pip install -r requirements.txt
```

## 3. Training the CNN

The CNN model can be trained in two methods:

Training on MNIST

Training on a custom dataset extracted from sudoku.com


```bash
python main.py
```

At startup, choose the 4th option:

```code
Choose recognition method:
1 - Tesseract
2 - Template Matching
3 - CNN
4 - Train CNN model
```

Then choose the training method:

```code
Choose TYPE :
1 - Sudoku grid
2 - MNIST
```

The MNIST method generates this file:
```code
digit_cnn.keras
```

The Sudoku grid method generates this file:
```code
digit_cnn_sudoku.keras
```

The project won't run if  these files are missing from the `/model` directory

## 4. Run the project

```bash
python main.py
```

At startup, choose the recognition method:

```code
Choose recognition method:
1 - Tesseract
2 - Template Matching
3 - CNN
4 - Train CNN model
```

MAKE SURE THAT THE SUDOKU GRID IS DISPLAYED ON THE MAIN SCREEN

The program will:

Capture the screen

Detect the Sudoku grid

Extract and preprocess the cells

Recognize digits

Solve the puzzle

Fill the solution automatically


## 5. Run the docker

Build the image :
```bash
docker build -t sudoku-solver .
```

Run the container:
```bash
docker run --rm sudoku-solver
```
Note that PyAutoGui does not work inside a headless container


