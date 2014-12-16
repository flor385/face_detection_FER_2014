"""
Defines common functions used
througout the project.
"""
import numpy as np
import pickle
import logging
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO

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
