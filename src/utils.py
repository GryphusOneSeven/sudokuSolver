import cv2

def show(title, img, scale=1.0):
    h, w = img.shape[:2]
    img_resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(title, img_resized)
    cv2.waitKey(0)
