import numpy
import cv2
import matplotlib.pyplot as plt

#ucitaj sliku
YIQ_average = cv2.imread('AverageFace//average_face_YIQ.jpg', 1)

array_Y = []
array_I = []
array_Q = []

for row in range (len(YIQ_average)):
    for col in range(len(YIQ_average[0])):
        array_Y.append(YIQ_average[row][col][0])
        array_I.append(YIQ_average[row][col][1])
        array_Q.append(YIQ_average[row][col][2])
            
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


