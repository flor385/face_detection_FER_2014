"""
Module that deals with face detection.
Defines a Detector class that detects
faces (rectangular areas) in an image.
Provides means for optimizing and evaluating
a detector on the FDDB database
"""
import logging
import util
import numpy as np
import histogram
import fddb
import scipy.spatial
import cv2
import os

log = logging.getLogger(__name__)

#   this path is used for writing out images during detection
#   it is for debugging purposes, these images are not necessary
#   in production
#   to disable this functionality set the path to None
DEBUG_DIR = "DETECTION_DEBUG"
if DEBUG_DIR is not None:
    if not os.path.exists(DEBUG_DIR):
        os.makedirs(DEBUG_DIR)


class FaceClassifier(object):

    """
    A face classifier object. Classifies an image
    on being a face or not based on it's distance
    from the given average face.

    Distance is calculated in grayscale space. The image
    being classified is padded and scaled to be of shape
    identical to the average face (which must be a square).

    Distance metric can be set, so can be classification
    treshold.
    """

    def __init__(self, avg_face, avg_face_mask, treshold, dist_calc):
        """
        Initializes the face classifier.

        :param avg_face: Average face image. Must be a
            square image in grayscale.

        :param avg_face_mask: Average face mask. A boolean array
            of shape same as avg_face. Has True on pixels where
            where the face is in the image (in average), False
            elsewhere. The purpose of it is to extract the face
            pixels from the background (by setting background
            pixels to 0), thus improving distance-based face
            classification.

        :param treshold: The maximum distance an image
            can have from the average image to be classified
            as a face.

        :param dist_calc: A distance calculation function.
            Must be a callable that takes two 1D numpy
            arrays and returns a scalar, such as the
            functions in scipy.spatial.distance
        """

        #   making sure the avg face is grayscale square
        assert avg_face.ndim == 2
        #   making sure that avg face and mask are of OK sizes
        assert avg_face.shape[0] == avg_face.shape[1]
        assert avg_face_mask.shape[0] == avg_face_mask.shape[1]
        assert avg_face_mask.shape[0] == avg_face.shape[0]

        #   distances are calculated with flattened images
        self.avg_face_shape = avg_face.shape
        self.avg_face = avg_face.flatten()
        self.avg_face_mask = avg_face_mask

        self.dist_calc = dist_calc

    def is_face(self, image_RGB, reject_too_small=True):
        """
        Classifies image_RGB as representing a face or not.

        :param image_RGB: An image in RGB color space, a
            numpy array of shape (height, width, RGB).

        :reject_too_small: If or not images that are too
            small (smaller then the average face) should
            be rejected immediately. Image is too small if
            both it's weight and height are smaller then
            average image's.
        """
        if reject_too_small:
            if image_RGB.shape[0] < self.avg_face_shape[0]:
                if image_RGB.shape[1] < self.avg_face_shape[0]:
                    return False

        #   convert image to grayscale
        image_gray = image_RGB.mean(axis=2).astype(image_RGB.dtype)

        #   ensure image is a square of appropriate dimensions
        image_gray = util.image_in_square_box(
            image_gray, self.avg_face_shape[0])

        #   apply the mask that zeroes the background
        image_gray[self.avg_face_mask] = 0

        #   flatten image
        image_flat = image_gray.flatten()

        #   calculate distance and return
        dist = self.dist_calc(image_flat, self.avg_face)
        return dist < self.treshold


class Detector(object):

    """
    A face detector object. Given an unconstrained image it
    detects faces in it and returns the bounding boxes (square shaped)
    for those faces.

    Based on skin color detection in YIQ color space and subsequent
    comprison to an average face. Both steps are parameterised.
    """

    def __init__(self, yiq_skin_ranges, face_classifier):
        """
        Constructor.

        :param yiq_skin_ranges: A numpy array of shape (3, 2).
            Each row corresponds to a channel in YIQ. Columns
            are upper and lower bounds for channel values that
            indicate skin. For a color in YIQ to be accepted
            as skin color it needs to satisfy all channels.

        :param face_classifier: A FaceClassifier object that
            does final classificaton of bouding boxes found
            by skin-color-detection.
        """
        assert yiq_skin_ranges.shape == (3, 2)

        self.yiq_skin_ranges = yiq_skin_ranges
        self.face_classifier = face_classifier

    def __bbox_clip(img, bbox):
        """
        Helper function that returns an image clip given bounding box info.
        """
        return img[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]]

    def detect(self, image_RGB):
        """
        Main detection function. Processes the given image, detects faces
        and returns a list of bouding boxes for found faces. Bounding boxes
        are squares. A bounding box is each box is an iterable of two points:
        box upper left coordinates and box lower right coordinates.
        Each point is an iterable of 2 ints that are point coordinates in numpy
        image representation.
        """

        #   convert image to YIQ
        image_YIQ = util.rgb_to_yiq(image_RGB)

        #   do skin color detection
        skin_pixels = (image_YIQ > self.yiq_skin_ranges[:, 0]) & \
            (image_YIQ < self.yiq_skin_ranges[:, 1])
        skin_pixels = skin_pixels.all(axis=2)

        #   store mask, if debugging is enabled
        if DEBUG_DIR is not None:
            #   write out one image in 100
            if np.random.randint(0, 99) == 0:
                nr = np.random.randint(0, 1e6)
                path = "{:06d}_orig.jpg".format(nr)
                cv2.imwrite(os.path.join(DEBUG_DIR, path), image_RGB)
                path = "{:06d}_mask.jpg".format(nr)
                cv2.imwrite(os.path.join(DEBUG_DIR, path),
                            skin_pixels.astype(np.uint8) * 255)

        #   do mask processing
        log.error("Implement mask processing")

        #   create bounding boxes
        #   each box is an iterable of two points: box upper left and
        #       box lower right.
        #   each point is an iterable of 2 coordinates in image (numpy style)
        bboxes = []
        log.error("Implement face bound-boxing")

        #   finally see what the face classifier says
        #   note that the classifier works on the grayscale image made from RGB
        image_gray = np.mean(image_RGB, axis=2).astype(image_RGB.dtype)
        bboxes = [b for b in bboxes if self.face_classifier.is_face(
            self.__bbox_clip(b, image_gray))]

        return bboxes


def evaluation():

    log.info("Face detection evaluation on FDDB")
    log.info("We'll be doing 10-fold cross-evaluation with "
             "embedded 9-fold cross-validation")

    #   defining the parameter space
    #   parameter space is a cartesian product (we do grid searching)
    #   on parameter values of all parameter types
    param_space = []
    for avg_size in [32, 64]:
        for dist_tsh in [scipy.spatial.distance.cosine]:
            for dist_mtr in [scipy.spatial.distance.euclidean]:
                p_set = (avg_size, dist_tsh, dist_mtr)
                param_space.append(p_set)

    #   outer fold for evaluation purposes
    all_folds = range(1, 11)
    for test_fold in all_folds:
        log.info("Testing on fold %d", test_fold)

        #   prepare the validation folds
        validation_folds = list(all_folds)
        validation_folds.remove(test_fold)

        #   try out detector parameters
        for param_set in param_space:

            #   unpack prameters from parameter space
            avg_face_size, face_dist_tsh, face_dist_metric = param_set

            #   validation loop
            for validation_fold in validation_folds:
                log.info("Validating on fold %d", validation_fold)

                #   prepare which are the training folds
                training_folds = list(validation_folds)
                training_folds.remove(validation_fold)

                #   fit detector parameters on training-folds
                skin_ranges = histogram.face_range(training_folds)
                avg_face = fddb.avg_face(training_folds, avg_face_size)
                avg_mask = fddb.avg_mask(training_folds, avg_face_size) > 127

                #   create face classifier and detector
                fc = FaceClassifier(
                    avg_face, avg_mask, face_dist_tsh, face_dist_metric)
                detector = Detector(skin_ranges, fc)

                #   evaluate detector on validation_fold
                for img_path in fddb.image_file_paths(validation_fold):
                    bboxes = detector.detect(cv2.imread(img_path, 1))

                    #   match bboxes to elipses
                    #   for eatch match calc intersection_size / union_size
                    #   for bboxes and elipses that are without their pair
                    #   add 0 to score and 1 to detection count
                    log.error("Implement box-elipse matching")

        #   fit detector with best performing parameters
        #   test on the testing fold
        log.error("Fit detector on best params for eval")

    #   report cross-validation results
    log.error("Report classification results")


def main():

    logging.basicConfig(level=logging.DEBUG)
    log.info("Detection main, will start detector evaluation")

    evaluation()


if __name__ == '__main__':
    main()
