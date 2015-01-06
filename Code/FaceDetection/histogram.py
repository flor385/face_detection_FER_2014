"""
Histogram equalization module.
Normalizes the brightness and increases the contrast of the image.
"""

import cv2
import pickle

EQ_IMAGES_PATH = "equal_face_images.p"


def equalize(image_path, equal_image_path):
    face_images = pickle.load(open(image_path, "rb"))
    equal_images = []

    #   equalized images are also represented as tuples
    #   (grayscale image representation, label)
    for image in face_images:
        equal_image = cv2.equalizeHist(image[0])
        equal_images.append((equal_image, image[1]))

    pickle.dump(equal_images, open(equal_image_path, "wb"))

equalize("face_images.p", EQ_IMAGES_PATH)