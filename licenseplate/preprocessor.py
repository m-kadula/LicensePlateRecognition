import numpy as np
from numpy.typing import NDArray
import cv2


def preprocess_identity(image: NDArray) -> NDArray:
    return image.copy()


def preprocess_black_on_white(image: NDArray) -> NDArray:
    image = image.copy()

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                               cv2.THRESH_BINARY, 11, 2)
    # sharpened = cv2.GaussianBlur(image, (0, 0), 3)
    # image = cv2.addWeighted(image, 1.5, sharpened, -0.5, 0)
    image = image[:, :, :2]
    image = 255 - np.max(image, axis=2)
    _, image = cv2.threshold(image, 150, 255, cv2.THRESH_BINARY)
    return image
