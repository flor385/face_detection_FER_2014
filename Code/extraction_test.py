import FaceExtraction
import os


image_path = "FDDB"
information_path = os.path.join("FDDB", "TrainingFaces", "FDDB-folds")
training_path = os.path.join("FDDB", "TrainingFaces")
average_path = "AverageFace"

FaceExtraction.generate_faces(image_path, information_path, training_path)
