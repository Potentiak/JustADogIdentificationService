# -*- coding: utf-8 -*-
"""
Created on Thu May 12 08:23:04 2022

@author: Killshot
"""

from os import listdir
from random import randint
from matplotlib.image import imread
import matplotlib.pyplot as plt

DATA_SET_PATH = 'C:\\Users\\dolacins\\Documents\\DATA\\Images'
breeds = listdir(DATA_SET_PATH)

# load image paths
breed_images_paths = []
for breed in breeds:
    path_to_breed = DATA_SET_PATH + '\\' + str(breed)
    breed_images_names = listdir(path_to_breed)
    path_for_breed_image_to_display = path_to_breed + '\\' + breed_images_names[randint(1, len(breed_images_names)) - 1]
    breed_images_paths.append(path_for_breed_image_to_display)


# load and display images in one figure
fig = plt.figure()

for index in range(120):
    breed_image = imread(breed_images_paths[index])
    fig.add_subplot(10, 12, index + 1)
    plt.imshow(breed_image)
    plt.axis('off')

plt.tight_layout()
plt.suptitle("All the dog breeds in stanford dogs dataset", y=1.001)  # #finetuning
plt.show()
