import cv2
import numpy as np

from src.features import LBP


def read_grayscale_image(path: str, width: int, height: int):
    im_grayscale = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im_resized = cv2.resize(im_grayscale, (width, height))
    return im_resized


def transform_and_show(path, radius: int, neighbors: int):
    image = read_grayscale_image(path, 128, 128)

    cv2.imwrite("Pred.png", image)

    transformed = np.array(LBP(image, radius, neighbors))

    cv2.imwrite("Po.png", transformed)
