import cv2
import numpy as np
import tensorflow as tf

class DigitRecognizer:
    def __init__(self, model_path="model/digit_cnn_sudoku.keras"):
        self.model = tf.keras.models.load_model(model_path)

    def preprocess(self, img):
        # img = cellule binaire
        img = cv2.resize(img, (28, 28))
        img = img.astype("float32") / 255.0
        img = img[..., None]  # (28, 28, 1)
        return img
    
    def is_empty_cell(self, cell):
        # cell doit être binaire (0 ou 255)
        white = cv2.countNonZero(cell)
        h, w = cell.shape

        return white < (h * w * 0.08)


    def predict(self, img):

        if self.is_empty_cell(img):
            print("empty cell")
            return 0

        x = self.preprocess(img)
        x = np.expand_dims(x, axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        print(int(np.argmax(probs)))
        return int(np.argmax(probs))
