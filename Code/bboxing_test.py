"""
A module for trying out mask-based
bounding box creation. Mask opening
and dilation is used to remove noise
and achieve coherent and compact areas
which are then boxed.
"""

import cv2
import numpy as np
from detection import DEBUG_DIR
import logging
import os
from scipy import ndimage

log = logging.getLogger(__name__)


def bboxes_from_mask(mask, img, img_name):

    # remove mask noise (small particles)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # now dilate remaining mask to merge areas
    dilated = cv2.dilate(opening, kernel, iterations=5)

    # do component labeling
    markers, component_count = ndimage.label(dilated)
    cv2.imwrite(img_name + "_mask6_markers.jpg",
                markers.astype(np.uint8) * (255 / max(1, component_count)))

    #   remove components too big or too small
    #   first calc all component sizes
    comp_sizes = np.array([(markers == c).sum() for c in range(
        1, component_count + 1)], dtype=float)
    #   normalize sizes with image size
    comp_sizes /= np.prod(mask.shape)

    #   remove compnents substantially smaller then average
    comp_size_avg = comp_sizes.mean()
    for comp_ind, size in enumerate(comp_sizes):
        size = size / comp_size_avg
        if size < 0.5:
            markers[markers == (comp_ind + 1)] = 0

    cv2.imwrite(img_name + "_mask7_markers2.jpg",
                markers.astype(np.uint8) * (255 / max(1, component_count)))

    #   for each of the remaining components create a bounding box
    bboxes = []
    for comp in range(1, component_count + 1):
        mask = markers == comp
        if mask.sum() == 0:
            continue
        vert_projection = np.arange(mask.shape[1])[mask.sum(axis=0) != 0]
        hori_projection = np.arange(mask.shape[0])[mask.sum(axis=1) != 0]
        bbox = np.zeros((2, 2), np.int)
        bbox[0][0] = hori_projection.min()
        bbox[0][1] = vert_projection.min()
        bbox[1][0] = hori_projection.max()
        bbox[1][1] = vert_projection.max()
        bboxes.append(bbox)

    #   draw the boxes over an image
    for bbox in bboxes:
        img[bbox[0][0]:bbox[1][0], [bbox[0][1], bbox[1][1]]] = (128, 255, 128)
        img[[bbox[0][0], bbox[1][0]], bbox[0][1]:bbox[1][1]] = (128, 255, 128)
    cv2.imwrite(img_name + "_mask8_image_bboxes.jpg", img)

    return bboxes


def prune():
    images = [os.path.join(DEBUG_DIR, f) for f in os.listdir(DEBUG_DIR)]
    images = [f for f in images if (f.find("_orig") == -1)]
    images = [f for f in images if (f.find("_mask.") == -1)]
    for image in images:
        log.info("Deleting %s", image)
        os.remove(image)


def main():

    logging.basicConfig(level=logging.DEBUG)
    log.info("BBoxing text.py:main()")

    prune()

    images = [os.path.join(DEBUG_DIR, f) for f in os.listdir(DEBUG_DIR)]
    images = [f for f in images if (f.find("_orig") != -1)]
    images = [f[:f.find("_orig")] for f in images]

    for img_name in images:
        log.info("Processing image %s", img_name)

        #   load image
        img = img_name + "_orig.jpg"
        log.debug("Loading %s", img)
        img = cv2.imread(img, 1)

        #   load mask
        mask = img_name + "_mask.jpg"
        log.debug("Loading %s", mask)
        mask = cv2.imread(mask, 0)

        #   get bboxes
        bboxes = bboxes_from_mask(mask, img, img_name)


if __name__ == "__main__":
    main()
