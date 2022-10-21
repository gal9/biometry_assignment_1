import cv2


def read_grayscale_image(path: str, width: int, height: int):
    im_grayscale = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    im_resized = cv2.resize(im_grayscale, (width, height))
    return im_resized
