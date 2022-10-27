import os
from turtle import distance
import cv2
import sys
import numpy as np
from skimage.feature import local_binary_pattern

from src.data import read_grayscale_image, transform_and_show
from src.features import flatten_image, LBP_interpolation, LBP_uniform, LBP_histogram
from src.recognition import euclidian_distance_metric_recognition, TP_rate_from_distances

def workflow(radius_neighbors_pairs=[], methode="LBP", width: int=128, height: int=128):
    data_location = "data/awe"

    processed = [[] for _ in radius_neighbors_pairs]

    c = 0

    # Loop through data directory
    for filename in os.listdir(data_location):
        f = os.path.join(data_location, filename)

        if os.path.isdir(f):
            processed_tmp = [[] for _ in radius_neighbors_pairs]

            for image_file in os.listdir(f):
                image_location = os.path.join(f, image_file)


                if(image_file.endswith(".png")):
                    # print(f"Reading image {image_location}", end="\r")
                    c += 1
                    print(f"Reading image {c}/1000; {image_location}", end="\r")

                    # Reading grayscale image and transforming it to 1D vector
                    image_grayscale = read_grayscale_image(image_location, width, height)
                    if(methode == "LBP"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(flatten_image(LBP_interpolation(image_grayscale, pair[0], pair[1], width, height)))
                    elif(methode == "LBP_uniform"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(flatten_image(LBP_uniform(image_grayscale, pair[0], pair[1], width, height)))
                    elif(methode == "LBP_histogram_8x8"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 8, 8, width, height))
                    elif(methode == "LBP_histogram_4x4"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 4, 4, width, height))
                    elif(methode == "LBP_histogram_16x16"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 16, 16, width, height))
                    elif(methode == "LBP_histogram_32x32"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(LBP_histogram(image_grayscale, pair[0], pair[1], 32, 32, width, height))
                    elif(methode == "sklearn_lbp"):
                        for i, pair in enumerate(radius_neighbors_pairs):
                            processed_tmp[i].append(flatten_image(local_binary_pattern(image_grayscale, R=pair[0], P=pair[1])))
                    elif(methode == "pixel_by_pixel"):
                        processed_tmp[0].append(flatten_image(image_grayscale))

            for i, pair in enumerate(radius_neighbors_pairs):
                processed[i].append(np.array(processed_tmp[i]))

    processed = np.array(processed)

    print()
    print("Results for method " + methode)

    with open('readme.txt', 'a') as f:
        f.write("Results for method " + methode + "\n")


    for i, pair in enumerate(radius_neighbors_pairs):
        #np.save(f"results/{methode}_{pair[0]}_{pair[1]}", processed[i])
        distances_euclidean = euclidian_distance_metric_recognition(processed[i], metric="euclidean")
        distances_cosin = euclidian_distance_metric_recognition(processed[i], metric="cosine")
        tp_rate_euclidean = TP_rate_from_distances(distances_euclidean)
        tp_rate_cosin = TP_rate_from_distances(distances_cosin)
        
        with open('readme.txt', 'a') as f:
            f.write(f"Radius: {pair[0]}; neighbors: {pair[1]} => euclidean: {tp_rate_euclidean}; cosine: {tp_rate_cosin} \n") # noqua
            f.flush()

        print(f"Radius: {pair[0]}; neighbors: {pair[1]} =>  euclidean: {tp_rate_euclidean}; cosine: {tp_rate_cosin} ")

    
