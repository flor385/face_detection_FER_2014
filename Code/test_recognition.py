# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 00:36:57 2015

@author: Edi
"""

import recognition
import os

def testOnTheDatabase(train_set_path, database_path) :
    BASE_IMG_DIR = train_set_path
    rec = recognition.RecognitionArh2(BASE_IMG_DIR)
    face_index_lst = readFacesOrder(os.path.join(train_set_path, "train_set.nfo"))
    print("TRAINED")
    folders = filter(lambda p : os.path.isdir(os.path.join(database_path, p)),\
    os.listdir(database_path))
    folders = map(lambda x : os.path.join(database_path, x), folders)
    succ_global = 0
    num_faces_global = 0
    for indfold in xrange(len(folders)) :
        folder = folders[indfold]
        files = map(lambda f : os.path.join(folder, f), os.listdir(folder))
        files = filter(lambda f : os.path.isfile(f) and \
        unicode.endswith(f, ".pgm") and not unicode.endswith(f, "Ambient.pgm"), \
        files)
        success = 0
        num_faces_global += len(files)
        for fil in files :
            simils = rec.get_similarities_for(fil, similarity_type="cosine")
            pos, sim = max(simils, key = lambda x : x[1])
            if (face_index_lst[pos] == indfold) :
                success += 1
        print("Face " + str(indfold) + ": " + str(success) + "/" + str(len(files)))
        succ_global += success
    print("Global precision: " + str(float(succ_global) / num_faces_global))
        
def readFacesOrder(train_set_info_path):
    f = open(train_set_info_path, "r")
    face_index_lst = []
    for line in f :
        line = line.strip()
        if line != "" :
            face_index_lst.append(int(line))
    f.close()
    return face_index_lst
        
if __name__ == "__main__":
    testOnTheDatabase("./FDDB/train_set".decode("utf-8"),\
    "./FDDB/CroppedYale".decode("utf-8"))