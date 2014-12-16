"""
Module that deals with face detection.
Defines a Detector class that detects
faces (rectangular areas) in an image.
Provides means for optimizing and evaluating
a detector on the FDDB database
"""
import logging
import util

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
