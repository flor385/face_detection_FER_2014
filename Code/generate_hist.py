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
                if current_image[row][col][0] != 0:
                    array_Y.append(current_image[row][col][0])
                if current_image[row][col][1] != 0:
                    array_I.append(current_image[row][col][1])
                if current_image[row][col][2] != 0:
                    array_Q.append(current_image[row][col][2])
            
    hist_Q, bin_edges_Q = numpy.histogram(array_Q, bins = range(256))

    #linije koje se mogu odkomentirati radi testiranja
    """
    plt.bar(bin_edges_Q[:-1], hist_Q, width = 1)
    plt.xlim(min(bin_edges_Q), max(bin_edges_Q))
    plt.show()   

    print hist_Q
    """

    hist_I, bin_edges_I = numpy.histogram(array_I, bins = range(256))

    #linije koje se mogu odkomentirati radi testiranja
    """
    plt.bar(bin_edges_I[:-1], hist_I, width = 1)
    plt.xlim(min(bin_edges_I), max(bin_edges_I))
    plt.show()
    print hist_I
    """

    return hist_I, hist_Q

#metoda koja je ista po funkcionalnosti kao test.py, ali omogucava definiranje tresholda
#direktno iz histograma, bez pogadanja
def generate_binary_image(image_path):
    image = cv2.imread(image_path, 1)
    YIQ_image = FaceExtraction.convert_RGB_to_YIG(image)
    hist_I, hist_Q = generate_IQ_histograms()
    #TODO omoguciti trazenje treshold vrijednosti te bojanje slike u skladu s njima
    
