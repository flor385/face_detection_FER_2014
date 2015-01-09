"""
Module that enters Yale Face Database from specified PATH to
defined data structure, equalizes their histogram using module histogram
and saves it as FACE_IMAGES_PATH. Images are saved as tuples
(grayscale image representation, label) in two files, gallery_images and
probe_images. Gallery set is a set of one representative image of every person, while
probe images are the ones those identity is being recognized.
"""

import os
import cv2
import pickle

DB_PATH = "..\\CroppedYale"
IMAGES_PATH = ""


def get_images(db_path, image_path):

    gallery_images = []
    probe_images = []

    #   reading images from db_path
    #   labels are subdirectory names, one person's images in each subdirectory
    for dirname, dirnames, filenames in os.walk(db_path):

        for subdir in dirnames:
            subdir_path = os.path.join(dirname, subdir)
            for filename in os.listdir(subdir_path):

                #   creating gallery and probe data set
                if filename.endswith(".pgm") and filename.find("Ambient") == -1:
                    image = cv2.imread(os.path.join(subdir_path, filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)

                    #   histogram equalization
                    image = cv2.equalizeHist(image)
                    label = str(subdir)

                    if filename.find("P00A+000E+00") == -1:
                        probe_images.append((image, label))
                    else:
                        gallery_images.append((image, label))

    pickle.dump(gallery_images, open(image_path + "gallery_images.p", "wb"))
    pickle.dump(probe_images, open(image_path + "probe_images.p", "wb"))

    return gallery_images, probe_images

get_images(DB_PATH, IMAGES_PATH)