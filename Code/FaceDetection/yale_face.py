"""
Module that enters Yale Face Database from specified PATH to
defined data structure and saves it as FACE_IMAGES_PATH.
Images are saved as tuples (grayscale image representation, label).
"""

import os
import cv2
import pickle

PATH = "..\\CroppedYale"
FACE_IMAGES_PATH = "face_images.p"


def get_images(db_path, image_path):

    face_images = []

    #   reading images from db_path
    #   labels are subdirectory names, one person's images in each subdirectory
    for dirname, dirnames, filenames in os.walk(db_path):

        for subdir in dirnames:
            subdir_path = os.path.join(dirname, subdir)
            for filename in os.listdir(subdir_path):
                if filename.endswith(".pgm"):
                    image = cv2.imread(os.path.join(subdir_path, filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)
                    label = str(subdir)
                    face_images.append((image, label))

    pickle.dump(face_images, open(image_path, "wb"))

get_images(PATH, FACE_IMAGES_PATH)