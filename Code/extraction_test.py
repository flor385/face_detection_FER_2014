import numpy
import matplotlib
import cv2
import math
from os import listdir
from os.path import isfile, join
import FaceExtraction


image_path = "FDDB"
information_path = "TrainingFaces//FDDB-folds"
training_path = "TrainingFaces"
average_path = "AverageFace"

FaceExtraction.generate_faces(image_path, information_path, training_path)



