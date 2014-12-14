import numpy
import cv2
import matplotlib.pyplot as plt
import os
import FaceExtraction
from os import listdir
from os.path import isfile, join

training_path = FaceExtraction.training_path

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

x = [max(array_Q)]
default_range = numpy.array((0.9, 1.1)) * x
            
hist, bin_edges = numpy.histogram(array_Q, bins = range(256))
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.show()   

print hist

hist, bin_edges = numpy.histogram(array_I, bins = range(256))
plt.bar(bin_edges[:-1], hist, width = 1)
plt.xlim(min(bin_edges), max(bin_edges))
plt.show()

print hist

