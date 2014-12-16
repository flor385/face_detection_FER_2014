"""
A module for generating histograms
of colors in the FDDB database.

Histograms are made for two classes:
face and not-face.

Histograms can be made
in the RGB or YIQ color spaces. It is
expected that the YIQ color space is more
suitable for color-based face detection.
"""
import fddb
import logging
import numpy as np
import cv2
import util
import matplotlib.pyplot as plt
import os

log = logging.getLogger(__name__)

#   root folder for histogram pickled data and image storage
HIST_ROOT = fddb.PATH_ROOT + "_histogram"
if not os.path.exists(HIST_ROOT):
    os.makedirs(HIST_ROOT)

#   these are the bin edges for histogram
#   again first dimension is Y, I or Q
#   note that Y falls in [0, 255] but
#   I and Q fall into [-127, 127] roughly
__bin_edges = np.zeros((3, 257), np.int)
__bin_edges[0, :] = np.arange(0, 257)
__bin_edges[1:, :] = np.arange(-128, 129)


def histograms(fold):
    """
    Generates YIQ component histograms for
    face and not-face parts of the images
    of the given fold(s) of the FDDB database.

    Returns a tuple (hist_face, hist_noface).

    :type fold: int or iterable of ints
    :param fold: When int: number of the fold of
        the FDDB database. When iterable: a number
        of folds for the FDDB database.
    """

    if not isinstance(fold, int):
        #   fold param is an iterable
        #   get individual fold histograms
        hists_face, hists_noface = zip(*[histograms(f) for f in fold])

        #   sum them up and return
        fold_count = len(hists_face)
        hist_face = sum(hists_face[1:], hists_face[0]) / fold_count
        hist_noface = sum(hists_noface[1:], hists_noface[0]) / fold_count
        return (hist_face, hist_noface)

    #   generate file name in which this fold's histograms are stored
    hist_file_name = os.path.join(
        HIST_ROOT, "fddb_YIQ_histogram_fold_{:02d}.pkl".format(fold))

    #   try to load and return pickled histogram data
    pickled_hist = util.try_pickle_load(hist_file_name)
    if pickled_hist is not None:
        return pickled_hist

    #   failed to load pickled data, create histograms

    #  prepare histograms
    #  first dimension indicates Y, I or Q,
    #  second dimension are bins
    hist_face = np.zeros((3, 256), np.int)
    hist_noface = np.zeros((3, 256), np.int)

    #   go through all the photos in the fold
    #   and their FDDB elipsis info (face annotations)
    for photo_path, elipses in fddb.image_elipses(fold).items():

        log.info("Processing photo %s", photo_path)

        #   load photo, convert to YIO
        log.debug("Loading photo")
        photo_RGB = cv2.imread(photo_path, 1)
        log.debug("Converting to YIQ")
        photo_YIQ = util.rgb_to_yiq(photo_RGB)

        #   create masks from elipses and OR them into one mask
        log.debug("Creating faces mask")
        photo_shape = photo_YIQ.shape[:2]
        mask_face = np.zeros(photo_shape, np.bool)
        for elipse in elipses:
            mask_face = mask_face | fddb.elipsis_mask_and_box(
                photo_shape, elipse)[0]

        mask_noface = np.logical_not(mask_face)

        #   add current image histograms to total histograms
        log.debug("Histogramming")
        for component_ind in range(3):
            hist_face[component_ind, :] += np.histogram(
                photo_YIQ[mask_face, component_ind],
                __bin_edges[component_ind]
            )[0]
            hist_noface[component_ind, :] += np.histogram(
                photo_YIQ[mask_noface, component_ind],
                __bin_edges[component_ind]
            )[0]

    #   normalize histograms
    hist_face = hist_face.astype(np.float) / hist_face[1, :].sum()
    hist_noface = hist_noface.astype(np.float) / hist_noface[1, :].sum()

    #   store histogram data for subsequent usage
    if not util.try_pickle_dump((hist_face, hist_noface), hist_file_name):
        raise "Failed to pickle histograms"

    return (hist_face, hist_noface)


def plot_hist(hist_face, hist_noface, show=True, file_name=None):
    """
    Plots the YIQ face and not-face histograms for comparison.

    :param hist_face: YIQ histograms of face pixels.
    :param hist_noface: YIQ histograms of not-face pixels.
    :param show: If the plot should be shown onscreen.
    :param file_name: Name of the file to store the plot in.
        If None, the plot image is not stored.
    """

    #   plot bounds (per component) that we know make sense
    bounds = [(0, 256), (-80, 80), (-50, 50)]

    #   start plotting
    plt.figure(figsize=(12, 9), dpi=72)

    #   iterate through the components (YIQ)
    for i in range(3):
        plt.subplot(2, 2, i + 1)
        plt.plot(__bin_edges[i][:-1],
                 hist_face[i], label="Face")
        plt.plot(__bin_edges[i][:-1],
                 hist_noface[i], label="Not face")
        plt.xlim(bounds[i])
        plt.xlabel("YIQ"[i] + " component")
        plt.legend()

    #   show plot or save to file
    if file_name is not None:
        plt.savefig(file_name, bbox_inches='tight')
    if show:
        plt.show()


def face_range(fold, treshold=0.2):
    """
    Calculates tresholds (ranges) for the face
    class for each of the YIQ components.
    Returns a numpy array of shape (3, 2) where
    first dimension indicates component and
    second is range start/end.

    :param fold: When int: number of the fold of
        the FDDB database. When iterable: a number
        of folds for the FDDB database.

    :param treshold: The minimum amount by which
        the face histogram needs to exceede the
        not-face histogram in face-range. Expressed
        as fraction of face component maximum.
    """
    hist_face, hist_noface = histograms(fold)
    diff = hist_face - hist_noface
    ranges = np.zeros((3, 2))

    #   iterate through the components
    for i in range(3):

        #   start at diff maximum
        range_start_ind = np.argmax(diff[i])
        range_end_ind = range_start_ind

        #   maximum value
        max = diff[i][range_start_ind]

        #   grow range both ways
        while diff[i][range_start_ind - 1] >= (max * treshold):
            range_start_ind -= 1
        while diff[i][range_end_ind + 1] >= (max * treshold):
            range_end_ind += 1

        #   store range
        ranges[i][0] = __bin_edges[i][range_start_ind]
        ranges[i][1] = __bin_edges[i][range_end_ind]

    return ranges


def main():
    """
    Main funtion. Generates and prints out
    the histograms on all the folds of FDDB.
    """
    logging.basicConfig(level=logging.DEBUG)
    log.info("Histogram test for FDDB")

    #  generate individual histograms
    for i in range(1, 11):
        img_file_path = os.path.join(
            HIST_ROOT, "fddb_YIQ_histogram_fold_{:02d}.pdf".format(i))
        hist_face, hist_noface = histograms(i)
        plot_hist(hist_face, hist_noface, show=False, file_name=img_file_path)

    #   show total histogram
    hist_face, hist_noface = histograms(range(1, 11))
    img_file_path = os.path.join(HIST_ROOT, "fddb_YIQ_histogram_all_folds.pdf")
    plot_hist(hist_face, hist_noface, show=True, file_name=img_file_path)


if __name__ == "__main__":
    main()
