# -*- coding: utf-8 -*-

import recognition
import recognition_train_set_extractor
import os
import pickle
import matplotlib.pyplot as plt

def testOnTheDatabase(train_set_path, database_path, equalize, simil_type) :
    """
    Learns the architecture 2 using the training set (folder with pictures and a
    description file) with train_set_path and tests it on the entire cropped database.
    For every person/face in the database the number of successful recognitions is
    printed. At the end the precision on the database is printed.
    """
    BASE_IMG_DIR = train_set_path
    #learn model
    rec = recognition.RecognitionArh2(BASE_IMG_DIR, equalize_hist = equalize)
    #read training set description
    face_index_lst = readFacesOrder(os.path.join(train_set_path, "train_set.nfo"))
    print("TRAINED")
    folders = filter(lambda p : os.path.isdir(os.path.join(database_path, p)),\
    os.listdir(database_path))
    folders = map(lambda x : os.path.join(database_path, x), folders)
    succ_global = 0
    num_faces_global = 0
    for indfold in xrange(len(folders)) :
        folder = folders[indfold]
        from_train = face_index_lst.count(indfold)
        files = map(lambda f : os.path.join(folder, f), os.listdir(folder))
        files = filter(lambda f : os.path.isfile(f) and \
        unicode.endswith(f, ".pgm") and not unicode.endswith(f, "Ambient.pgm"), \
        files)
        success = 0
        num_faces_global += len(files) - from_train
        for fil in files :
            simils = rec.get_similarities_for(fil, similarity_type=simil_type)
            pos, sim = max(simils, key = lambda x : x[1])
            if (face_index_lst[pos] == indfold) :
                success += 1
        print("Face " + str(indfold) + ": " + str(success - from_train) + "/" +\
        str(len(files) - from_train))
        succ_global += success - from_train
    prec = float(succ_global) / num_faces_global
    print("Global precision: " + str(succ_global) + "/" + str(num_faces_global) + \
    " = " + str(prec))
    return prec
        
def readFacesOrder(train_set_info_path):
    """
    Helper function for reading the training set description. The description
    is really a list of integers which represent to which person does the
    face on that specific index in the set belong.
    """
    f = open(train_set_info_path, "r")
    face_index_lst = []
    for line in f :
        line = line.strip()
        if line != "" :
            face_index_lst.append(int(line))
    f.close()
    return face_index_lst
        
if __name__ == "__main__":
    lst_cos_std = []
    lst_cos_nstd = []
    lst_euc_std = []
    res_path = "./FDDB/results.p"
    if (not os.path.exists(res_path)) :
        for numFaces in xrange(1, 16) :
            recognition_train_set_extractor.extractTrainSamples(\
            "./FDDB/CroppedYale".decode('utf-8'),\
            numFaces, "./FDDB/train_set")
            f1 = testOnTheDatabase("./FDDB/train_set".decode("utf-8"),\
            "./FDDB/CroppedYale".decode("utf-8"), True, "cosine")
            f2 = testOnTheDatabase("./FDDB/train_set".decode("utf-8"),\
            "./FDDB/CroppedYale".decode("utf-8"), False, "cosine")
            f3 = testOnTheDatabase("./FDDB/train_set".decode("utf-8"),\
            "./FDDB/CroppedYale".decode("utf-8"), True, "euclid")
            lst_cos_std.append(f1)
            lst_cos_nstd.append(f2)
            lst_euc_std.append(f3)
        resFile = open(res_path, "w")
        pickle.dump((lst_cos_std, lst_cos_nstd, lst_euc_std), resFile)
        resFile.close()
    resFile = open(res_path, "r")
    lst_cos_std, lst_cos_nstd, lst_euc_std = pickle.load(resFile)
    resFile.close()
    print(str(lst_cos_std))
    print(str(lst_cos_nstd))
    print(str(lst_euc_std))
    xaxis = range(1, 16)
    plt.figure(0)
    plt.plot(xaxis, lst_cos_std, label="equalized")
    plt.plot(xaxis, lst_cos_nstd, label="non-equalized")
    plt.axis([1, 15, 0, 1])
    plt.xticks(range(1, 16))
    plt.yticks([x / 10. for x in range(11)])
    plt.legend(loc=0)
    plt.show()
    plt.figure(1)
    plt.plot(xaxis, lst_cos_std, label="cosine")
    plt.plot(xaxis, lst_euc_std, label="euclidean")
    plt.axis([1, 15, 0, 1])
    plt.xticks(range(1, 16))
    plt.yticks([x / 10. for x in range(11)])
    plt.legend(loc=0)
    plt.show()