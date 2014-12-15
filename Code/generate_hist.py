import numpy
import cv2
import matplotlib.pyplot as plt
import os
import FaceExtraction
from os import listdir
from os.path import isfile, join

training_path = FaceExtraction.training_path

#generira histograme za I i Q komponente te ih vraca tim redom
def generate_IQ_histograms():
    face_images = [ f for f in listdir(training_path) if isfile(join(training_path,f)) ]

    array_Y = [] 
    array_I = []
    array_Q = [] 

    for image_path in face_images:
        current_image = cv2.imread(training_path + os.sep + image_path, 1)
        current_image = FaceExtraction.convert_RGB_to_YIQ(current_image)
        for row in range (len(current_image)):
            for col in range(len(current_image[0])):
                if current_image[row][col][0] != 0 and current_image[row][col][0] != 255:
                    array_Y.append(current_image[row][col][0])
                if current_image[row][col][1] != 0 and current_image[row][col][1] != 255:
                    array_I.append(current_image[row][col][1])
                if current_image[row][col][2] != 0 and current_image[row][col][2] != 255:
                    array_Q.append(current_image[row][col][2])
            
    hist_Q, bin_edges_Q = numpy.histogram(array_Q, bins = 256)

    #linije koje se mogu odkomentirati radi testiranja
    """
    plt.bar(bin_edges_Q[:-1], hist_Q, width = 1)
    plt.xlim(min(bin_edges_Q), max(bin_edges_Q))
    plt.show()   
    
    print hist_Q
    """

    hist_I, bin_edges_I = numpy.histogram(array_I, bins = 256)

    #linije koje se mogu odkomentirati radi testiranja
    """
    plt.bar(bin_edges_I[:-1], hist_I, width = 1)
    plt.xlim(min(bin_edges_I), max(bin_edges_I))
    plt.show()
    print hist_I
    """

    return hist_I, hist_Q

#metoda koja je ista po funkcionalnosti kao test.py, ali omogucava definiranje thresholda
#direktno iz histograma, bez pogadanja
def generate_binary_image(image_path):
    image = cv2.imread(image_path, 1)
    YIQ_image = FaceExtraction.convert_RGB_to_YIQ(image)
    threshold_image = numpy.zeros((len(YIQ_image), len(YIQ_image[0]), 3), numpy.uint8)
    hist_I, hist_Q = generate_IQ_histograms()
    threshold_I = numpy.argsort(hist_I)[-1]
    threshold_Q = numpy.argsort(hist_Q)[-1]

    for row in range(0, len(YIQ_image)):
        for col in range(0, len(YIQ_image[0])):
            if YIQ_image[row][col][1] in (0, 255) or YIQ_image[row][col][2] in (0, 255):
                continue
            if YIQ_image[row][col][1] >= threshold_I - 10 and YIQ_image[row][col][1] <= threshold_I + 10 and YIQ_image[row][col][2] >= threshold_Q - 10 and YIQ_image[row][col][2] <= threshold_Q + 10:
                threshold_image[row][col] = 255
            else:
                threshold_image[row][col] = 0 

    #linije koje se mogu odkomentirati radi testiranja
    """
    cv2.imshow('image', threshold_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

#metoda koja se ne koristi trenutno, isprobavana kao alternativa racunanju thresholda    
def get_top_20(array):
    threshold = [0] * len(array)
    sorted_by_index = numpy.argsort(array)
    counter = 0
    for i in reversed(sorted_by_index):
        threshold[i] = 1
        counter += 1
        if counter == 20:
            break

    return threshold
