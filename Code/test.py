import numpy
import cv2
import FaceExtraction

image = cv2.imread('FDDB//2002//08//11//big//img_591.jpg', 1)

YIQ_image = FaceExtraction.convert_RGB_to_YIQ(image)

threshold_image = numpy.zeros((len(YIQ_image), len(YIQ_image[0]), 3), numpy.uint8)

for row in range(0, len(YIQ_image)):
    for col in range(0, len(YIQ_image[0])):
        #Pogadao sam threshold vrijednosti pomocu histograma generiranog sa generate_hist.py
        if YIQ_image[row][col][1] > 225 and YIQ_image[row][col][1] < 245 and YIQ_image[row][col][2] < 15 and YIQ_image[row][col][2] > 2:
            threshold_image[row][col][0] = 255
            threshold_image[row][col][1] = 255
            threshold_image[row][col][2] = 255
        else:
            threshold_image[row][col][0] = 0
            threshold_image[row][col][1] = 0
            threshold_image[row][col][2] = 0

cv2.imshow('image', threshold_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
