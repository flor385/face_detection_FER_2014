"""
A module for FDDB database handling.
Provides functions for loading of
images file names and elipse information.
"""
import os
import logging
import re
import numpy as np
import collections
import math
import cv2
import util

#   root folder of the FDDB database
PATH_ROOT = "FDDB"

log = logging.getLogger(__name__)


def image_file_paths(fold):
    """
    Returns all the image file paths belonging
    to a given fold.

    :param fold: int indicating which fold is
        desired. In [1, 10] range.
    """

    #   fold image paths are stored in a text file
    path = os.path.join(PATH_ROOT, "FDDB-fold-{:02d}.txt".format(fold))
    with open(path) as f:
        return [os.path.join(PATH_ROOT, line.strip()) for line in f]


def image_elipses(fold):
    """
    Returns a dictionary in which keys are file
    paths of images belonging to the fold, and
    values are lists of face-elipses for that image.
    Each face-elipse is a numpy array of format:

    [major_axis_radius minor_axis_radius angle center_x center_y]

    :param fold: int indicating which fold is
        desired. In [1, 10] range.
    """

    #   the dictionary we will return
    r_val = collections.OrderedDict()

    #   fold image paths are stored in a text file
    path = os.path.join(
        PATH_ROOT, "FDDB-fold-{:02d}-ellipseList.txt".format(fold))
    with open(path) as f:

        #   read file lines, keep only the stripped non-empty ones
        lines = [l.strip() for l in f if len(l.strip()) > 0]

        #   parse lines
        #   lines are split into groups, per file
        line_iterator = iter(lines)
        while True:

            #   image name (no extension) is the first line in group
            img_name = next(line_iterator, None)
            if img_name is None:
                break

            img_path = os.path.join(PATH_ROOT, img_name + ".jpg")
            r_val[img_path] = []

            #   second line is the number of faces in the photo
            face_count = int(line_iterator.next())

            #   following are elipses (1 per line) for all faces
            for _ in range(face_count):
                elipsis_info = re.split("\\s+", line_iterator.next())
                elipsis_info = elipsis_info[:-1]
                elipsis = np.array([float(s) for s in elipsis_info])
                r_val[img_path].append(elipsis)

    return r_val


def elipsis_mask_and_box(img_shape, elipsis_info):
    """
    Generates a boolaen mask of given shape
    that has True values for pixels within
    the elipsis, and False without.

    Returns the mask and a bounding box of the elipsis
    in image_shape coordinates.

    Elipsis info is given in FDDB format:
    [major_axis_radius minor_axis_radius angle center_x center_y]
    """

    #   unpacking elipsis info
    major_axis, minor_axis, angle, center_x, center_y = elipsis_info

    def check_if_in_ellipse(point_x, point_y):
        """
        Calculates if a given point is within the elipsis.
        """
        cosa = math.cos(angle)
        sina = math.sin(angle)
        dd = minor_axis * minor_axis
        DD = major_axis * major_axis

        a = math.pow(
            cosa * (point_x - center_x) + sina * (point_y - center_y), 2)
        b = math.pow(
            sina * (point_x - center_x) - cosa * (point_y - center_y), 2)
        ellipse = (a / DD) + (b / dd)

        if ellipse <= 1:
            return True
        else:
            return False

    #   mask for the given shape
    r_val = np.zeros(img_shape, np.bool)
    #   bounding box info
    #   bounding box stores two points, upper left and lower right
    #   for both points (i, j) coordinates (numpy orientation) are stored
    iinfo = np.iinfo(np.int32)
    bbox = np.array([[iinfo.max, iinfo.max], [iinfo.min, iinfo.min]], np.int32)

    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            if not check_if_in_ellipse(j, i):
                continue

            r_val[i][j] = True

            #   update upper left corner
            bbox[0][0] = min(bbox[0][0], i)
            bbox[0][1] = min(bbox[0][1], j)
            #   update lower right corner
            bbox[1][0] = max(bbox[1][0], i)
            bbox[1][1] = max(bbox[1][1], j)

    return r_val, bbox


def faces(fold):
    """
    Retrieves a list of face images. Images are numpy arrays
    of (img_height, img_width, RGB) shape. The images represent
    the clipped and masked face images from the given fold
    of the FDDB database.

    :param fold: int indicating which fold is
        desired. In [1, 10] range.
    """

    #   generate file name in which this fold's face images are stored
    faces_file_name = os.path.join(
        PATH_ROOT, "fddb_facesonly_fold_{:02d}.zip".format(fold))

    #   try to load and return pickled data
    face_images = util.try_pickle_load(faces_file_name, zip=True)
    if face_images is not None:
        return face_images

    #   resulting face images
    #   each image is a numpy array of RGB components of
    #   (img_height, img_width, 3) shape
    face_images = []

    #   go through all the photos in the fold
    #   and their FDDB elipsis info (face annotations)
    for photo_path, elipses in image_elipses(fold).items():

        log.info("Processing photo %s", photo_path)

        #   load photo
        log.debug("Loading photo")
        photo_RGB = cv2.imread(photo_path, 1)
        photo_shape = photo_RGB.shape[:2]

        #   for each elipse info get mask and bbox
        for elipse in elipses:

            log.debug("Calculating mask and bounds")
            mask, bbox = elipsis_mask_and_box(photo_shape, elipse)

            #   apply the bounding box
            log.debug("Applying mask and bounds")
            face_img = np.array(
                photo_RGB[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1], :])

            #   apply the mask
            face_mask = mask[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]]
            face_img[np.logical_not(face_mask), :] = 0

            #   store the image
            face_images.append(face_img)

    #   store image data for subsequent usage
    if not util.try_pickle_dump(face_images, faces_file_name, zip=True):
        raise "Failed to pickle face images"


def main():
    """
    Main function of this modules.
    Just prints out some tests to see if
    the functions are working.
    """

    logging.basicConfig(level=logging.DEBUG)
    log.info("Testing FDDB loading")

    faces(4)
    raise "Delete this and previous line"

    file_paths = image_file_paths(1)
    log.info("First ten files of the first fold:")
    for f in file_paths[:10]:
        log.info("\t%s", f)

    elipses = image_elipses(10)
    log.info("First ten elipses of the tenth fold:")
    for file_path, elipse_list in elipses.items()[:10]:
        log.info("\t%s", file_path)
        for elipse in elipse_list:
            log.info("\t%s", elipse)

if __name__ == "__main__":
    main()
