import numpy
import cv2
import FaceExtraction
import os
import histogram
import util

def test_histogram_py():
    ranges = histogram.face_range(1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_390.jpg'), 1)
    YIQ_image = util.rgb_to_yiq(image)

    threshold_image = numpy.zeros((len(YIQ_image), len(YIQ_image[0]), 3), numpy.uint8)

    
    for row in range(0, len(YIQ_image)):
        for col in range(0, len(YIQ_image[0])):
            if YIQ_image[row][col][1] >= ranges[1][0] and YIQ_image[row][col][1] <= ranges[1][1] and YIQ_image[row][col][2] >= ranges[2][0] and YIQ_image[row][col][2] <= ranges[2][1]:
                threshold_image[row][col] = 255
            else:
                threshold_image[row][col] = 0

    cv2.imshow('image', threshold_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_default_hist():    
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_591.jpg'), 1)

    YIQ_image = FaceExtraction.convert_RGB_to_YIQ(image)

    threshold_image = numpy.zeros((len(YIQ_image), len(YIQ_image[0]), 3), numpy.uint8)

    for row in range(0, len(YIQ_image)):
        for col in range(0, len(YIQ_image[0])):
            #Pogadao sam threshold vrijednosti pomocu histograma generiranog sa generate_hist.py
            if YIQ_image[row][col][1] > 225 and YIQ_image[row][col][1] < 250 and YIQ_image[row][col][2] < 15 and YIQ_image[row][col][2] > 2:
                threshold_image[row][col] = 255
            else:
                threshold_image[row][col] = 0

    cv2.imshow('image', threshold_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
