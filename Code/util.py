"""
Defines common functions used
througout the project.
"""
import numpy as np
import pickle
import logging
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import cv2

log = logging.getLogger(__name__)

__YIQ_TO_RGB = np.matrix(
    "0.299 0.596 0.212; 0.587 -0.275 -0.523; 0.114 -0.321 0.311")


def rgb_to_yiq(rgb_data):
    """
    Takes image data in RGB space and
    generates and returns the same image
    in YIQ space.

    :param rgb_data: A numpy array that needs
        to be a matrix of (pixel_count, 3) shape
        or a tensor of (width, height, 3) shape.
        YIQ data is returned in the same
        shape as the given rgb_data.
    """

    #   remember the shape of data
    rgb_shape = rgb_data.shape
    assert rgb_shape[-1] == 3

    #   for matrix multiplication, reshape original
    rgb_flat = rgb_data.reshape((np.prod(rgb_shape[:-1]), 3))

    #   calculate YIQ data
    yiq_data = np.dot(rgb_flat, __YIQ_TO_RGB)

    #   return yiq in original shape
    return np.array(yiq_data).reshape(rgb_shape)


def image_in_square_box(img, side_size=None, padding_value=0):
    """
    Fits the image into a square box, the original image
    being centered in it. If side_size param is given, the image
    is fitted into a square box of desired size.

    :param img: The image to fit.
    :param side_size: The desired size of square side. If None,
        an image is only padded into a square.
    :param padding_value: The value to pad the image with,
        default is 0.
    """

    #   image padding that works on images regardless of
    #   how many channels (last dimension) they have
    #   first we need to reshape image so it has 3 dimensions
    shape = img.shape
    img = img.reshape(shape[0], shape[1], np.prod(shape) / np.prod(shape[:2]))
    #   now determine the padding amounts to make the image a square
    dim = max(shape[:2])
    pad_up = (dim - img.shape[0]) / 2
    pad_down = dim - img.shape[0] - pad_up
    pad_left = (dim - img.shape[1]) / 2
    pad_right = dim - img.shape[1] - pad_left
    #   now do actual padding
    img = np.pad(img, ((pad_up, pad_down), (pad_left, pad_right), (0, 0)),
                 'constant', constant_values=padding_value)

    #   if no size is given, return the padded image
    if side_size is None:
        return img

    return cv2.resize(img, (side_size, side_size))


def bbox_overlap(bb1, bb2):
    """
    Calculates and returns the number of pixels
    that the two bounding boxes have in common.

    Both boxes are iterables of two points, where
    first is the upper left point and the second
    is lower right. Each
    point is an iterable of two coordinates (ints).

    :param bb1: First bbox.
    :param bb2: First bbox.
    """
    x_overlap = max(0, min(bb1[1][1], bb2[1][1]) - max(bb1[0][1], bb2[0][1]))
    y_overlap = max(0, min(bb1[1][0], bb2[1][0]) - max(bb1[0][0], bb2[0][0]))
    return x_overlap * y_overlap


def bbox_for_mask(mask):
    """
    For a boolean mask returns the bounding box (rectangular)
    of the True values. The bounding box encircles all the
    True values.

    :param mask: A numpy array of booleans.
    """
    vert_projection = np.arange(mask.shape[1])[mask.sum(axis=0) != 0]
    hori_projection = np.arange(mask.shape[0])[mask.sum(axis=1) != 0]
    bbox = np.zeros((2, 2), np.int)
    bbox[0][0] = hori_projection.min()
    bbox[0][1] = vert_projection.min()
    bbox[1][0] = hori_projection.max()
    bbox[1][1] = vert_projection.max()

    return bbox


def try_pickle_load(file_name, zip=None):
    """
    Tries to load pickled data from a file with
    the given name. If unsuccesful, returns None.
    Can compress using Zip.

    :param file_name: File path/name.
    :param zip: If or not the file should be zipped.
        If None, determined from file name.
    """
    if zip is None:
        zip = file_name.lower().endswith("zip")

    try:
        if zip:
            file = ZipFile(file_name, 'r')
            entry = file.namelist()[0]
            data = pickle.load(BytesIO(file.read(entry)))
        else:
            file = open(file_name, "rb")
            data = pickle.load(file)
        log.info('Succesfully loaded pickle %s', file_name)
        return data
    except IOError:
        log.info('Failed to load pickle %s', file_name)
        return None
    finally:
        if 'file' in locals():
            file.close()


def try_pickle_dump(data, file_name, zip=None, entry_name="Data.pkl"):
    """
    Pickles given data tp the given file name.
    Returns True if succesful, False otherwise.

    :param data: The object to pickle.
    :param file_name: Name of file to pickle to.
    :param zip: If or not the file should be zipped.
        If None, determined from file name.
    :param entry_name: If zipping, the name to be used
        for the ZIP entry.
    """
    if zip is None:
        zip = file_name.lower().endswith("zip")

    try:
        log.info('Attempting to pickle data to %s', file_name)
        if zip:
            file = ZipFile(file_name, 'w', ZIP_DEFLATED)
            file.writestr(entry_name, pickle.dumps(data))
        else:
            pickle.dump(data, open(file_name, "wb"))
        return True
    except IOError:
        log.info('Failed to pickle data to %s', file_name)
        return False
    finally:
        if 'file' in locals():
            file.close()


def main():
    logging.basicConfig(level=logging.DEBUG)
    log.info("Testing rect overlap")

    def print_bb_info(bbox):
        log.info("BBox in (x,y): upper-left: (%d, %d), lower-right: (%d, %d)",
                 bbox[0][1], bbox[0][0], bbox[1][1], bbox[1][0])

    bb1 = [[10, 10], [20, 20]]
    bb2 = [[15, 15], [25, 25]]
    print_bb_info(bb1)
    print_bb_info(bb2)
    log.info("Overlap: %d\n\n", bbox_overlap(bb1, bb2))

    bb1 = [[10, 10], [20, 20]]
    bb2 = [[20, 20], [25, 25]]
    print_bb_info(bb1)
    print_bb_info(bb2)
    log.info("Overlap: %d\n\n", bbox_overlap(bb1, bb2))

    bb1 = [[10, 10], [20, 20]]
    bb2 = [[5, 5], [15, 15]]
    print_bb_info(bb1)
    print_bb_info(bb2)
    log.info("Overlap: %d\n\n", bbox_overlap(bb1, bb2))

    bb1 = [[10, 10], [20, 20]]
    bb2 = [[5, 10], [15, 10]]
    print_bb_info(bb1)
    print_bb_info(bb2)
    log.info("Overlap: %d\n\n", bbox_overlap(bb1, bb2))

    bb1 = [[10, 10], [20, 20]]
    bb2 = [[-10, 0], [100, 5]]
    print_bb_info(bb1)
    print_bb_info(bb2)
    log.info("Overlap: %d\n\n", bbox_overlap(bb1, bb2))

if __name__ == "__main__":
    main()
