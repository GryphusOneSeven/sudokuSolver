import cv2
import numpy as np
import os

class TemplateMatcher:
    def __init__(self, template_dir="dataset/templates"):
        self.templates = {}
        for digit in range(10):
            path = os.path.join(template_dir, f"{digit}.png")
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            self.templates[digit] = img

    def preprocess(self, img):
        img = cv2.resize(img, (40, 40))
        return img

    def predict(self, cell):
        cell = self.preprocess(cell)

        best_digit = 0
        best_score = -1

        for digit, tmpl in self.templates.items():
            tmpl = cv2.resize(tmpl, (40, 40))
            res = cv2.matchTemplate(cell, tmpl, cv2.TM_CCOEFF_NORMED)
            score = res[0][0]

            if score > best_score:
                best_score = score
                best_digit = digit

        return best_digit
