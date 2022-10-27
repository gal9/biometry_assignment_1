import numpy as np
import math
import cv2


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


def LBP(image, radius: int, neighbors: int, width: int, height: int):
    coordinates = get_coordinates(radius, neighbors, True)

    values = []
    for row_i, row in enumerate(image):
        row_values = []
        for pixel_i, pixel in enumerate(row):
            # If pixel in question is on the edge set value to 0
            if(row_i < radius or pixel_i < radius or row_i+radius >= width or pixel_i+radius >= height):
                row_values.append(0)
            else:
                neighbor_coordinates = [(row_i+y, pixel_i+x) for x, y in coordinates]
                row_values.append(LBP_value(image, pixel, neighbor_coordinates))

        values.append(row_values)

    return np.array(values)


def LBP_value_interpolation(image, center_value, coordinates):
    lbp = 0
    for i, point in enumerate(coordinates):
        # First coordinate represents row (y axis) and second represents column (x axis)

        pixle_value = 0
        pixle_value += image[math.ceil(point[0])][math.ceil(point[1])]
        pixle_value += image[math.ceil(point[0])][math.floor(point[1])]
        pixle_value += image[math.floor(point[0])][math.ceil(point[1])]
        pixle_value += image[math.floor(point[0])][math.floor(point[1])]
        pixle_value /= 4

        if(center_value <= pixle_value):
            lbp += 2**i

    return lbp


def LBP_interpolation(image, radius: int, neighbors: int, width: int, height: int):
    coordinates = get_coordinates(radius, neighbors, False)

    values = []
    for row_i, row in enumerate(image):
        row_values = []
        for pixel_i, pixel in enumerate(row):
            # If pixel in question is on the edge set value to 0
            if(row_i < radius or pixel_i < radius or row_i+radius >= height or pixel_i+radius >= width):
                row_values.append(0)
            else:
                neighbor_coordinates = [(row_i+y, pixel_i+x) for x, y in coordinates]
                row_values.append(LBP_value_interpolation(image, pixel, neighbor_coordinates))

        values.append(row_values)

    return np.array(values)


def LBP_histogram(image, radius: int, neighbors: int, columns: int, rows: int, width: int, height: int):
    LBP_picture = LBP_interpolation(image, radius, neighbors, width, height)

    height = len(image)
    width = len(image[0])

    grid_height = int(height/rows)
    grid_width = int(width/columns)

    final_hist = np.array([])

    for column in range(columns):
        for row in range(rows):
            subarray = LBP_picture[row*grid_height:(row+1)*grid_height,
                column*grid_width:(column+1)*grid_width]

            hist = np.histogram(subarray, bins=range(257))

            final_hist = np.concatenate((final_hist, hist[0]), axis=0)

    return final_hist
            


lookup_table = [0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
        14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
        58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
        58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
        58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
        58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
        58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
        58,58,58,50,51,52,58,53,54,55,56,57]


def LBP_uniform(image, radius:int, neighbors: int, width: int, height: int):
    LBP_picture = LBP_interpolation(image, radius, neighbors, width, height)

    for row_i, row in enumerate(LBP_picture):
        for pixel_i, pixel in enumerate(row):
            LBP_picture[row_i][pixel_i] = lookup_table[pixel]

    return LBP_picture
