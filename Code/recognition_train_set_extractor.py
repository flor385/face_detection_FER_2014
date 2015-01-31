# -*- coding: utf-8 -*-

import os
import shutil

def extractTrainSamples(dbPath, numSamplesPerFace, outDir) :
    """
    For every person in the database takes numSamplesPerFace images and
    copies them in the outDir (the directory of the training set). Every k-th
    image is taken where k = (num of images) / numSamplesPerFace. A training
    set info is also generated with filename 'train_set.nfo' which contains
    the index of the person an image in the training set sorted by file names
    refers to (used for testing purposes).
    """
    folders = filter(lambda p : os.path.isdir(os.path.join(dbPath, p)), os.listdir(dbPath))
    #deletes the folder and starts fresh
    if (not os.path.isdir(outDir)) :
        os.mkdir(outDir)
    else :
        shutil.rmtree(outDir)
        os.mkdir(outDir)
    pers_ind = 0
    file_face_belong = []
    fl_indices = indicesToTake(numSamplesPerFace)
    for fo in folders :
        fold = os.path.join(dbPath, fo)
        files = filter(lambda f : os.path.isfile(os.path.join(fold, f)) and \
        unicode.endswith(f, ".pgm") and not unicode.endswith(f, "Ambient.pgm"), \
        os.listdir(fold))
        indices = map(lambda x : int(round(x * (len(files) - 1))), fl_indices)
        for ind in indices :
            f = files[ind]
            file_face_belong.append((f, pers_ind))
            fout = os.path.join(outDir, f)
            fin = os.path.join(dbPath, fo, f)
            shutil.copy2(fin, fout)
        pers_ind += 1
    file_face_belong.sort(key = lambda x : x[0])
    f = open(os.path.join(outDir, "train_set.nfo"), "w")
    for face, index in file_face_belong :
        f.write(str(index)+"\n")
    f.close()
    
def indicesToTake(numFaces) :
    if (numFaces == 1) :
        return [0.0]
    elif (numFaces == 2) :
        return [0.0, 1.0]
        
    fl_indices = [0.0, 1.0]
    faces_took = 2
    curr_gap = 0.5
    prev_gap = 1.0
    while (faces_took < numFaces) :
        curr_num = curr_gap
        while (curr_num < 1.0):
            fl_indices.append(curr_num)
            curr_num += prev_gap
            faces_took += 1
            if (faces_took == numFaces):
                break
        prev_gap = curr_gap
        curr_gap /= 2
    fl_indices.sort()
    return fl_indices

if __name__ == "__main__":
    extractTrainSamples("./FDDB/CroppedYale".decode('utf-8'),\
    11, "./FDDB/train_set")