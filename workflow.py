import os

from src.data import read_grayscale_image
from src.features import flatten_image
from src.recognition import euclidian_distance_recognition


images_grayscale = []
images_flattened = []

data_location = "data/awe"

# Loop through data directory
for filename in os.listdir(data_location):
    f = os.path.join(data_location, filename)

    if os.path.isdir(f):
        grayscalles_tmp = []
        flattened_tmp = []

        for image_file in os.listdir(f):
            image_location = os.path.join(f, image_file)

            if(image_file.endswith(".png")):
                print(f"Reading image {image_location}", end="\r")

                # Reading grayscale image and transforming it to 1D vector
                image_grayscale = read_grayscale_image(image_location, 128, 128)
                image_flattened = flatten_image(image_grayscale)

                grayscalles_tmp.append(image_grayscale)
                flattened_tmp.append(image_flattened)

        images_grayscale.append(grayscalles_tmp)
        images_flattened.append(flattened_tmp)


TP = 0
c = 0
number_of_people = len(images_flattened)
number_of_samples = len(images_flattened)*len(images_flattened[0])

for persone_index, samples in enumerate(images_flattened):
    # print(f"Persone {persone_index+1}/{number_of_people}", end="\r")
    for sample_i, sample in enumerate(samples):
        c += 1
        print(f"Sample {c}/{number_of_samples}", end="\r")
        persone_recognised_i, image_recognised_i = euclidian_distance_recognition(sample, images_flattened, persone_index, sample_i) #noqa

        # Test if image was correctly classfied
        if(persone_index == persone_recognised_i):
            # print(persone_recognised_i)
            TP += 1

print()

print(TP/number_of_samples)