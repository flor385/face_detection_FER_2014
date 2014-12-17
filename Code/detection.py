"""
Module that deals with face detection.
Defines a Detector class that detects
faces (rectangular areas) in an image.
Provides means for optimizing and evaluating
a detector on the FDDB database
"""
import logging
import util
import os
import fddb
import cv2
import numpy as np

log = logging.getLogger(__name__)


class Detector(object):

    def __init__(self, yiq_skin_ranges, avg_face, avg_face_treshold):
        assert yiq_skin_ranges.shape == (3, 2)

        self.yiq_skin_ranges = yiq_skin_ranges
        self.avg_face = avg_face
        self.avg_face_treshold = avg_face_treshold

    def detect(self, image_RGB):

        #   convert image to YIQ
        image_YIQ = util.rgb_to_yiq(image_RGB)

        #   do skin color detection
        skin_pixels = (image_YIQ > self.yiq_skin_ranges[:, 0]) & \
            (image_YIQ < self.yiq_skin_ranges[:, 1])
        skin_pixels = skin_pixels.all(axis=2)

        #   do mask processing
        raise "Implement this"

        #   create bounding boxes
        raise "Implement this"

        #   compare bounding boxes to average face


#   root folder for average faces
AVG_FACE_ROOT = fddb.PATH_ROOT + "_avg_faces"
if not os.path.exists(AVG_FACE_ROOT):
    os.makedirs(AVG_FACE_ROOT)


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

    #   file name used to cache the result
    if isinstance(fold, int):
        file_name = "avg_face_{:02d}_size_{:d}.png".format(fold, size)
    else:
        fold_string = "folds_[" + ", ".join([str(f) for f in fold]) + "]"
        file_name = "avg_face_{:s}_size_{:d}.png".format(fold_string, size)
    file_name = os.path.join(AVG_FACE_ROOT, file_name)

    #   if given file exists, load and return it
    if os.path.isfile(file_name):
        return cv2.imread(file_name, 1)

    if isinstance(fold, int):
        #   load fold faces and filter out the too-small ones
        faces = fddb.faces(fold)
        raise "TODO: Filter out too small (any dimension lesser then size)"

        raise "TODO: Convert to grayscale"
        raise "TODO: Add padding to all the faces to make them squares"
        raise "TODO: Scale all the square faces to (size, size) shape"
        raise "TODO: Calulate average face"

    else:
        #   need to generate the average face
        #   for multiple folds calculate the average of individual folds
        #   we assume that folds are of similar sizes so averaging is OK
        avgs = np.array([avg_face(f, size) for f in fold])
        result = avgs.mean(axis=0)

    #   store the result
    cv2.imwrite(file_name, result)

    return result


def evaluation():

    log.info("Face detection evaluation on FDDB")
    log.info("We'll be doing 10-fold cross-evaluation with "
             "embedded 9-fold cross-validation")

    #   outer fold for evaluation purposes
    all_folds = range(1, 11)
    for test_fold in all_folds:
        log.info("Testing on fold %d", test_fold)

        #   prepare the validation folds
        validation_folds = list(all_folds)
        validation_folds.remove(test_fold)

        #   try out detector parameters
        for param_set in param_space:

            #   validation loop
            for validation_fold in validation_folds:
                log.info("Validating on fold %d", validation_fold)

                #   prepare which are the training folds
                training_folds = list(validation_folds)
                training_folds.remove(validation_fold)

                #   fit detector on training-folds

                #   evaluate detector on validation_fold

        #   fit detector with best performing parameters
        #   test on the testing fold

    #   report cross-validation results


def main():

    logging.basicConfig(level=logging.INFO)
    log.info("Detection main, will start detector evaluation")

    evaluation()


if __name__ == '__main__':
    main()
