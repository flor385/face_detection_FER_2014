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
from scipy import ndimage
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
        self.treshold = treshold

    def is_face(self, image_gray, reject_too_small=True):
        """
        Classifies image_gray as representing a face or not.

        :param image_gray: An image in grayscale color space, a
            numpy array of shape (height, width).

        :reject_too_small: If or not images that are too
            small (smaller then the average face) should
            be rejected immediately. Image is too small if
            both it's weight and height are smaller then
            average image's.
        """
        if reject_too_small:
            if image_gray.shape[0] < self.avg_face_shape[0]:
                if image_gray.shape[1] < self.avg_face_shape[1]:
                    log.debug("Face too small %r < %r",
                              image_gray.shape, self.avg_face_shape)
                    return False

        #   ensure image is a square of appropriate dimensions
        image_gray = util.image_in_square_box(
            image_gray, self.avg_face_shape[0])

        #   apply the mask that zeroes the background
        image_gray[self.avg_face_mask] = 0

        #   flatten image
        image_flat = image_gray.flatten()

        #   calculate distance and return
        dist = self.dist_calc(image_flat, self.avg_face)
        log.debug("Face dist from average: %.2f", dist)
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

    def __bbox_clip(self, img, bbox):
        """
        Helper function that returns an image clip given bounding box info.
        """
        return img[bbox[0][0]:bbox[1][0], bbox[0][1]:bbox[1][1]]

    def __bboxes_for_mask(self, mask):

        # remove mask noise (small particles) and then dillate
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        dilated = cv2.dilate(opening, kernel, iterations=5)

        # do component labeling
        markers, component_count = ndimage.label(dilated)

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

        #   for each of the remaining components create a bounding box
        masks = [markers == c for c in range(1, component_count + 1)]
        return [util.bbox_for_mask(m) for m in masks if m.sum() > 0]

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
        skin_pixels = skin_pixels.all(axis=2).astype(np.uint8) * 255

        #   get bounding boxes for the skin pixel mask
        bboxes = self.__bboxes_for_mask(skin_pixels)

        #   finally see what the face classifier says
        #   note that the classifier works on the grayscale image made from RGB
        image_gray = np.mean(image_RGB, axis=2).astype(image_RGB.dtype)
        bboxes_filtered = [b for b in bboxes if self.face_classifier.is_face(
            self.__bbox_clip(image_gray, b))]

        #   if debugging, store images for reviewing
        if (DEBUG_DIR is not None) & (np.random.randint(0, 25) == 0):
            nr = np.random.randint(0, 1e6)
            path = "{:06d}_0_orig.jpg".format(nr)
            cv2.imwrite(os.path.join(DEBUG_DIR, path), image_RGB)
            path = "{:06d}_1_mask.jpg".format(nr)
            cv2.imwrite(os.path.join(DEBUG_DIR, path), skin_pixels)

            #   draw the boxes over an image
            red = (128, 128, 255)
            im = np.array(image_RGB)
            for bbox in bboxes:
                im[bbox[0][0]:bbox[1][0], [bbox[0][1], bbox[1][1]]] = red
                im[[bbox[0][0], bbox[1][0]], bbox[0][1]:bbox[1][1]] = red
            path = "{:06d}_2_bboxes.jpg".format(nr)
            cv2.imwrite(os.path.join(DEBUG_DIR, path), im)

            #   draw the faces over an image
            green = (128, 255, 128)
            im = np.array(image_RGB)
            for bbox in bboxes_filtered:
                im[bbox[0][0]:bbox[1][0], [bbox[0][1], bbox[1][1]]] = green
                im[[bbox[0][0], bbox[1][0]], bbox[0][1]:bbox[1][1]] = green
            path = "{:06d}_3_faces.jpg".format(nr)
            cv2.imwrite(os.path.join(DEBUG_DIR, path), im)

        return bboxes_filtered


def match_score(bboxes_detected, bboxes_true, masks):
    """
    Calculates the score for given detection parameters.
    Returns the ratio between all intersetcions and
    all unions.

    :param bboxes_detected: Bounding boxes of faces
        detected in an image.
    :param bboxes_true: Bounding boxes of annotated
        faces. Only serve for matching purposes.
    :param masks: Boolean masks of the whole image.
    """

    #   sum of intersections and unions
    union_total = 0.0
    intersection_total = 0.0

    #   match detected bboxes to known truths
    #   bboxes_matched is a list of bboxes corresponding
    bboxes_matched = []
    #   to bboxes_true
    for bbox_true in bboxes_true:

        #   for current bbox_true find best match
        overlaps = [util.bbox_overlap(bbox_true, b) for b in bboxes_detected]
        if len(overlaps) > 0:
            best_bbox = bboxes_detected[np.argmax(overlaps)]
        else:
            best_bbox = None

        #   remember the best match and remove from list
        bboxes_matched.append(best_bbox)
        if best_bbox is not None:
            bboxes_detected = [
                b for b in bboxes_detected if not np.array_equal(b, best_bbox)]

    #   now for matched boxes calculated metric
    for mask, bbox in zip(masks, bboxes_matched):

        #   calculate true/detection intersection
        if bbox is None:
            intersection = 0
        else:
            intersection = mask[bbox[0][0]:bbox[1][0],
                                bbox[0][1]:bbox[1][1]].sum()

        #   calcualte truth/detection union
        union = mask.sum()
        if bbox is not None:
            union += (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])
        union -= intersection

        #   append the metrics to total scores
        union_total += union
        intersection_total += intersection

    #   also add non-matched detected to scoring
    for bb in bboxes_detected:
        union_total += (bbox[1][0] - bbox[0][0]) * (bbox[1][1] - bbox[0][1])

    return intersection_total / union_total


def detector_for_params(folds, param_set):
    """
    Returns a detector fitted on the given
    FDDB fold with the given parameter set.
    Parameter set is a tuple consiting of
    (average_face_size, face_dist_treshold,
        face_dist_metric).
    """

    avg_face_size, face_dist_tsh, face_dist_metric = param_set

    #   fit detector parameters on training-folds
    skin_ranges = histogram.face_range(folds)
    avg_face = fddb.avg_face(folds, avg_face_size)
    avg_mask = fddb.avg_mask(folds, avg_face_size) > 127

    #   create face classifier and detector
    fc = FaceClassifier(
        avg_face, avg_mask, face_dist_tsh, face_dist_metric)
    return Detector(skin_ranges, fc)


def evaluation():

    log.info("Face detection evaluation on FDDB")
    log.info("We'll be doing 10-fold cross-evaluation with "
             "embedded 9-fold cross-validation")

    #   defining the parameter space
    #   parameter space is a cartesian product (we do grid searching)
    #   on parameter values of all parameter types
    param_space = []
    for avg_size in [32]:
        for dist_tsh in [8.0]:
            for dist_mtr in [scipy.spatial.distance.euclidean]:
                p_set = (avg_size, dist_tsh, dist_mtr)
                param_space.append(p_set)

    def eval_detector(detector, folds):
        """
        A helper function that returns detection
        scoring on given FDDB folds.
        """
        scores = []
        for img_path, (masks, bboxes_true) in \
                fddb.image_face_masks_bboxes(folds).items():

            #   run the image through the detector and score it
            bboxes_detected = detector.detect(cv2.imread(img_path, 1))
            scores.append(match_score(bboxes_detected, bboxes_true, masks))

        return np.mean(scores)

    #   outer fold for evaluation purposes
    all_folds = range(1, 11)
    eval_scores = []
    for test_fold in all_folds:
        log.info("Testing on fold %d", test_fold)

        #   prepare the validation folds
        validation_folds = list(all_folds)
        validation_folds.remove(test_fold)

        #   a dict of scores for param sets
        param_set_score_dict = {}

        #   try out detector parameters
        for param_set in param_space:

            #   validation loop
            validation_fold_scores = []
            for validation_fold in validation_folds:
                log.info("Validating on fold %d", validation_fold)

                #   prepare which are the training folds
                training_folds = list(validation_folds)
                training_folds.remove(validation_fold)

                #   fit detector parameters on training-folds
                detector = detector_for_params(training_folds, param_set)

                #   and evaluate it on the current validation fold
                validation_fold_scores.append(
                    eval_detector(detector, validation_fold))

            #   remember total score for current param set
            param_set_score_dict[param_set] = np.mean(validation_fold_scores)

        #   fit detector with best performing parameters
        #   test on the testing fold
        best_params = param_set_score_dict.keys()[
            np.argmax(param_set_score_dict.values())]
        log.info("Best params for fold %d: %r", test_fold, best_params)

        #   do final fold evaluation
        best_detector = detector_for_params(validation_folds, best_params)
        score = eval_detector(best_detector, test_fold)
        eval_scores.append(score)
        log.info("Eval score on fold %d: %.2f", test_fold, score)

    #   report cross-validation results
    log.info("\n\nFinal evaluation results:")
    log.info("Best scores on all folds: %r", eval_scores)
    log.info("Average score: %.2f, +- %.2f",
             np.mean(eval_scores), np.std(eval_scores))


def main():

    logging.basicConfig(level=logging.INFO)
    log.info("Detection main, will start detector evaluation")

    evaluation()


if __name__ == '__main__':
    main()
