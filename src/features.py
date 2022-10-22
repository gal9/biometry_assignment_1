import numpy as np
import math
import cv2

from src.data import read_grayscale_image


def flatten_image(image):
    return np.array(image).flatten()


def get_coordinates(radius: int, neighbors: int, round_coords: bool):
    coordinates = []

    for p in range(neighbors):
        coordinates.append((-math.sin((p*2*math.pi)/neighbors)*radius,
                           math.cos((p*2*math.pi)/neighbors)*radius))

    if(round_coords):
        coordinates = [(round(x), round(y)) for x, y in coordinates]

    return coordinates


def LBP_value(image, center_value, coordinates):
    lbp = 0
    for i, point in enumerate(coordinates):
        # First coordinate represents row (y axis) and second represents column (x axis)
        pixel_value = image[point[0]][point[1]]

        if(center_value <= pixel_value):
            lbp += 2**i

    return lbp


def LBP(image, radius: int, neighbors: int):
    coordinates = get_coordinates(radius, neighbors, True)

    values = []
    for row_i, row in enumerate(image):
        row_values = []
        for pixel_i, pixel in enumerate(row):
            # If pixel in question is on the edge set value to 0
            if(row_i < radius or pixel_i < radius or row_i+radius >= 128 or pixel_i+radius >= 128):
                row_values.append(0)
            else:
                neighbor_coordinates = [(row_i+y, pixel_i+x) for x, y in coordinates]
                row_values.append(LBP_value(image, pixel, neighbor_coordinates))

        values.append(row_values)

    return np.array(values)


def transform_and_show(path, radius: int, neighbors: int):
    image = read_grayscale_image(path, 128, 128)

    cv2.imshow("Pred", image)


def LBP1_1_8(image):
    # values = []
    for row_i, row in enumerate(image):
        row_values = []
        for pixel_i, pixel in enumerate(row):
            # If pixel in question is on the edge set value to 0
            if(row_i == 0 or pixel_i == 0):
                row_values.append(0)
            else:
                pass
