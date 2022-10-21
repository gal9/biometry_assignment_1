import sys
from sklearn.metrics import pairwise_distances


def euclidian_distance_recognition(target_image_flattened, images_flattened,
                                   target_persone_i: int,
                                   target_sample_i: int):
    closest_sample = None
    closest_persone = None
    closest_distance = sys.float_info.max

    for persone_i, samples in enumerate(images_flattened):
        for sample_i, sample in enumerate(samples):
            if(persone_i != target_persone_i or sample_i != target_sample_i):
                distance = pairwise_distances([sample-target_image_flattened],
                                              metric="euclidean")
                if(distance < closest_distance):
                    closest_distance = distance
                    closest_persone = persone_i
                    closest_sample = sample_i

    return closest_persone, closest_sample
