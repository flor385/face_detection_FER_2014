"""
Module that prepares Yale Face Database for classification. Db is
separated into 2 data sets, gallery with representative images of
every person and probe with all other images.
"""

import os
import cv2
import pickle

DB_PATH = "..\\CroppedYale"
RESULT_PATH = "..\\CroppedYaleDatasets"


def prepare_yale_db(db_path, result_path):

    gallery_images = []
    probe_images = []

    #   reading images from db_path
    #   labels are subdirectory names, one person's images in each subdirectory
    for dirname, dirnames, filenames in os.walk(db_path):

        gallery_images_path = os.path.join(result_path, "GalleryImages")
        probe_images_path = os.path.join(result_path, "ProbeImages")

        if not os.path.exists(gallery_images_path):
            os.makedirs(gallery_images_path)

        if not os.path.exists(probe_images_path):
            os.makedirs(probe_images_path)

        for subdir in dirnames:
            subdir_path = os.path.join(dirname, subdir)
            for filename in os.listdir(subdir_path):

                #   creating gallery and probe data set
                if filename.endswith(".pgm") and filename.find("Ambient") == -1:
                    image = cv2.imread(os.path.join(subdir_path, filename), cv2.CV_LOAD_IMAGE_GRAYSCALE)

                    #   histogram equalization
                    #image = cv2.equalizeHist(image)

                    if filename.find("P00A+000E+00") == -1:
                        cv2.imwrite(os.path.join(probe_images_path, filename), image)
                    else:
                        cv2.imwrite(os.path.join(gallery_images_path, filename), image)

prepare_yale_db(DB_PATH, RESULT_PATH)