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
#   TODO: change this to "FDDB" for the real deal
PATH_ROOT = "FDDB"

#   root folder for average faces
AVG_FACE_ROOT = PATH_ROOT + "_avg_faces"
if not os.path.exists(AVG_FACE_ROOT):
    os.makedirs(AVG_FACE_ROOT)

#   root folder for face masks
FACE_MASK_ROOT = PATH_ROOT + "_face_masks"
if not os.path.exists(FACE_MASK_ROOT):
    os.makedirs(FACE_MASK_ROOT)

#   root folder for face masks
FACE_ONLY_ROOT = PATH_ROOT + "_face_only"
if not os.path.exists(FACE_ONLY_ROOT):
    os.makedirs(FACE_ONLY_ROOT)

log = logging.getLogger(__name__)


def __elipsis_mask_and_box(img_shape, elipsis_info):
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
        return [os.path.join(PATH_ROOT, line.strip() + ".jpg") for line in f]


def image_elipses(fold):
    """
    Returns a dictionary in which keys are file
    paths of images belonging to the fold, and
    values are lists of face-elipses for that image.
    The returned dictionary is ordered the same way
    the elisis info file is.

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


def image_face_masks_bboxes(fold):
    """
    Returns a dictionary in which keys are file
    paths of images belonging to the fold.
    Values are tuples (masks, bboxes) where "masks"
    are lists of face-elipse booleam masks for that image
    and "bboxes" are bounding box info for that image.
    The returned dictionary is ordered the same way
    the elisis info file is.
    """
    log.info("Retrieving image masks for fold %s", str(fold))

    #   file name of the cached version
    masks_file_name = os.path.join(
        FACE_MASK_ROOT, "fddb_face_masks_fold{:02d}.zip".format(fold))

    #   try to load and return pickled data
    masks = util.try_pickle_load(masks_file_name, zip=True)
    if masks is not None:
        return masks

    #   there is no pickled version, we need to create the masks
    masks_dict = collections.OrderedDict()

    for photo_path, elipses in image_elipses(fold).items():
        log.info("Processing photo %s", photo_path)

        #   load photo
        log.debug("Loading photo")
        photo_RGB = cv2.imread(photo_path, 1)
        photo_shape = photo_RGB.shape[:2]

        #   for each elipse info get mask and bbox, and store them
        #   first prepare the numpy arrays in which they are stored
        masks = np.zeros(
            (len(elipses), photo_shape[0], photo_shape[1]), dtype=np.bool)
        bboxes = np.zeros((len(elipses), 2, 2), dtype=np.int32)
        #   then out those arrays into the dict
        masks_dict[photo_path] = (masks, bboxes)
        #   and then fill up the arrays with real data
        for elipse_ind, elipse in enumerate(elipses):

            log.debug("Calculating mask and bounds")
            mask, bbox = __elipsis_mask_and_box(photo_shape, elipse)
            masks[elipse_ind] = mask
            bboxes[elipse_ind] = bbox

    #   store image data for subsequent usage
    if not util.try_pickle_dump(masks_dict, masks_file_name, zip=True):
        raise "Failed to pickle face masks"

    return masks_dict


def faces(fold):
    """
    Retrieves a list of face images. Images are numpy arrays
    of (img_height, img_width, RGB) shape. The images represent
    the clipped and masked face images from the given fold
    of the FDDB database.

    :param fold: int indicating which fold is
        desired. In [1, 10] range.
    """
    log.info("Retrieving face images for fold %s", str(fold))

    #   generate file name in which this fold's face images are stored
    faces_file_name = os.path.join(
        FACE_ONLY_ROOT, "fddb_facesonly_fold_{:02d}.zip".format(fold))

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
    for photo_path, (masks, bboxes) in image_face_masks_bboxes(fold).items():

        log.info("Processing photo %s", photo_path)

        #   load photo
        log.debug("Loading photo")
        photo_RGB = cv2.imread(photo_path, 1)

        #   for each elipse info get mask and bbox
        for mask, bbox in zip(masks, bboxes):

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

    return face_images


def avg_face(fold, size):
    """
    Caluclates the average face for the given fold(s).
    The resulting average face if of shape
    (size, size, 1), in grayscale. Face is centered in
    the square, aspect-ratio of original faces is retained.

    :param fold: int or iterable of ints. Indicates the
        fold(s) for which the average face is sought.

    :param size: int, indicates the desired size of both
        dimensions of the resulting average face.
    """
    log.info("Retrieving average face for fold %s and size %d",
             str(fold), size)

    #   file name used to cache the result
    if isinstance(fold, int):
        file_name = "avg_face_{:02d}_size_{:d}.png".format(fold, size)
    else:
        fold_string = "folds_(" + ",".join([str(f) for f in fold]) + ")"
        file_name = "avg_face_{:s}_size_{:d}.png".format(fold_string, size)
    file_name = os.path.join(AVG_FACE_ROOT, file_name)

    #   if given file exists, load and return it
    if os.path.isfile(file_name):
        return cv2.imread(file_name, 0)

    if isinstance(fold, int):
        #   load fold faces and filter out the too-small ones
        fold_faces = [f for f in faces(fold) if (
            f.shape[0] >= size) & (f.shape[1] >= size)]

        #   converting to grayscale
        fold_faces = [f.mean(axis=2).astype(f.dtype) for f in fold_faces]

        #   pad images into squares and resize to desired
        fold_faces = [util.image_in_square_box(f, size) for f in fold_faces]

        #   return mean face
        result = np.mean(fold_faces, axis=0).astype(fold_faces[0].dtype)

    else:
        #   need to generate the average face
        #   for multiple folds calculate the average of individual folds
        #   we assume that folds are of similar sizes so averaging is OK
        #   need to do mean with floats, to prevent accumulator overflow
        #   then after convert to original type
        avgs = [avg_face(f, size) for f in fold]
        result = np.mean(avgs, axis=0).astype(avgs[0].dtype)

    #   store the result
    cv2.imwrite(file_name, result)

    return result


def avg_mask(fold, size):
    """
    Caluclates the average mask for the given fold(s).
    The resulting average mask if of shape
    (size, size, 1), in grayscale. Mask is centered in
    the square, aspect-ratio of original masks is retained.

    :param fold: int or iterable of ints. Indicates the
        fold(s) for which the average face is sought.

    :param size: int, indicates the desired size of both
        dimensions of the resulting average mask.
    """
    log.info("Retrieving average mask for fold %s and size %d",
             str(fold), size)

    #   file name used to cache the result
    if isinstance(fold, int):
        file_name = "avg_mask_{:02d}_size_{:d}.png".format(fold, size)
    else:
        fold_string = "folds_(" + ",".join([str(f) for f in fold]) + ")"
        file_name = "avg_mask_{:s}_size_{:d}.png".format(fold_string, size)
    file_name = os.path.join(AVG_FACE_ROOT, file_name)

    #   if given file exists, load and return it
    if os.path.isfile(file_name):
        return cv2.imread(file_name, 0)

    if isinstance(fold, int):
        #   load fold masks and filter out the too-small ones
        fold_masks = []
        for _, (masks, bboxes) in image_face_masks_bboxes(fold).items():
            for mask, bbox in zip(masks, bboxes):
                #   check if bbox of sufficient size
                if (bbox[1][0] - bbox[0][0]) < size:
                    if (bbox[1][1] - bbox[0][1]) < size:
                        continue

                #   clip out the mask and append to all the fold masks
                face_mask = mask[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]]
                fold_masks.append(face_mask)

        #   convert masks to grayscale
        max_val = np.iinfo(np.uint8).max
        fold_masks = [(f.astype(np.uint8) * max_val) for f in fold_masks]

        #   pad masks into squares and resize to desired
        fold_masks = [util.image_in_square_box(f, size) for f in fold_masks]

        #   return mean mask
        result = np.mean(fold_masks, axis=0).astype(fold_masks[0].dtype)

    else:
        #   for multiple folds calculate the average of individual folds
        #   we assume that folds are of similar sizes so averaging is OK
        #   need to do mean with floats, to prevent accumulator overflow
        #   then after convert to original type
        avgs = [avg_mask(f, size) for f in fold]
        result = np.mean(avgs, axis=0).astype(avgs[0].dtype)

    #   store the result
    cv2.imwrite(file_name, result)

    return result


def main():
    """
    Main function of this modules.
    Just prints out some tests to see if
    the functions are working.
    """

    logging.basicConfig(level=logging.DEBUG)
    log.info("Testing FDDB loading")

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

    avg_mask(1, 32)
    avg_face(1, 32)

if __name__ == "__main__":
    main()
