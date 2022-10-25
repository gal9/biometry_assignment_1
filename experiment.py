import os
import cv2
import sys
import numpy as np

from src.data import read_grayscale_image, transform_and_show
from src.features import flatten_image, LBP_interpolation, LBP_uniform, LBP_histogram
from src.recognition import euclidian_distance_metric_recognition, TP_rate_from_distances
from src.workflow import workflow

workflow([(1,8), (2,8), (2, 12), (3,8), (3,16), (4, 8), (8, 16)], methode="LBP")


workflow([(1,8), (2,8), (2, 12), (3,8), (3,16), (4, 8), (8, 16)], methode="LBP_uniform")

workflow([(1,8), (2,8), (2, 12), (3,8), (3,16), (4, 8), (8, 16)], methode="LBP_histogram_8x8")

workflow([(1,8), (2,8), (2, 12), (3,8), (3,16), (4, 8), (8, 16)], methode="LBP_histogram_4x4")

workflow([(1,8), (2,8), (2, 12), (3,8), (3,16), (4, 8), (8, 16)], methode="LBP_histogram_16x16")

workflow([(1,8), (2,8), (2, 12), (3,8), (3,16), (4, 8), (8, 16)], methode="LBP_histogram_32x32")