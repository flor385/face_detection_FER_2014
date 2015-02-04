import os
import cv2
import numpy as np
from sklearn.decomposition import FastICA, PCA
from scipy.spatial.distance import cosine, euclidean


class RecognitionArh2 :
    """
    Class RecognitionArh2 is used for training the face recognizer (the training
    is done during the class initialization) and querying about similarities
    between a new picture and a stored train set.
    """
    
    def __init__(self, images_training_path, approximate_dim_num = 4096, equalize_hist = True) :
        """
        Used for train set initialization. Takes a path to the directory where
        the initial faces for training are written and an approximate number of features
        the images will be resized to.
        """
        
        #picture reading, resizing, flattening and normalization
        self.__equalize_hist = equalize_hist
        self.__images_training = images_training_path
        self.__approximate_dim_num = approximate_dim_num
        self.__face_index_lst = self.__readFacesOrder(os.path.join(images_training_path, "train_set.nfo"))
        picture_paths = os.listdir(images_training_path)
        picture_paths = filter(lambda x : \
            os.path.isfile(os.path.join(images_training_path, x)) \
            and unicode.endswith(x, ".pgm"), picture_paths)
        picture_paths.sort()
        flattened_images = []
        sizes = []
        for f in picture_paths:
            path = os.path.join(images_training_path, f)
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = self.__resize_img(img, approximate_dim_num)
            if (self.__equalize_hist) :
                img = cv2.equalizeHist(img)
            sizes.append(img.shape)
            flattened = np.array(img.flatten()) / 255.0 #normalization in a [0-1] range
            flattened_images.append(flattened)
             
        img_observations = np.array(flattened_images)
        self.__pixel_means = np.mean(img_observations, axis=0)
        img_observations -= self.__pixel_means #pixel centering
        self.__create_image_obser(img_observations)
        #self.__write_pca_images(os.path.join(images_training_path, 'test/'), sizes)
        #self.__write_basis_images(os.path.join(images_training_path, 'test/'), sizes)
        #self.__write_reconstructed_images(os.path.join(images_training_path, 'test/'), sizes)
        
        
    def __resize_img(self, image, num_pix):
        """
        Helper function for resizing a image while maintaining the image ratio. 
        """
        sizex, sizey = image.shape
        gc = self.__gcd(sizex, sizey)
        sizex /= gc
        sizey /= gc
        r = int(np.round(np.sqrt(num_pix / np.multiply(sizex, sizey))))
        return cv2.resize(image, (r * sizey, r * sizex))
        
    def __gcd(self, a, b) :
        """
        Greatest common divisor implementation for determining the image ratio.
        """
        while (b != 0) :
            rem = a % b
            a = b
            b = rem
        return a
        
    def __create_image_obser(self, image_observations) :
        """
        Creation of a space in which the images will be compared (learning stage).
        Firstly PCA is applied in order to reduce the number of features in the
        images. Reduction is done so that 99% of measured variance is covered.
        
        After that, ICA is performed on the coefficients calculated by transforming
        (reducing) the face images with PCA. From the learned ICA components
        basis_images (vectors), original images coefficients and transformation
        for new comming images are extracted.
        """
        pca = PCA()
        pca.fit(image_observations)
        sum = 0
        components_to_take = 0
        for ratio in pca.explained_variance_ratio_:
            components_to_take += 1
            sum += ratio
            if (sum > 0.99):
                break 
        print("PCA reduces the number of dimensions to: " + str(components_to_take))
        pca = PCA(whiten=True, n_components=components_to_take)
        self.__transformed_images = pca.fit_transform(image_observations)
        self.__transformed_images_mean = np.mean(self.__transformed_images, axis=0)
        self.__transformed_images -= self.__transformed_images_mean
        self.__pca = pca
        
        
        ica = FastICA(whiten=True, max_iter=100000)
        self.__original_images_repres = ica.fit_transform(self.__transformed_images)
        self.__basis_images = ica.mixing_.T
        self.__transformation = ica.components_
        #print(U_)
    
        
    def __write_basis_images(self, path, sizes) :
        """
        Helper (debug) method for writing the learned basis images to a folder.
        """
        for row_ind in range(len(self.__basis_images)) :
            img = self.__pca.inverse_transform(self.__basis_images[row_ind])
            S_min, S_max = np.amin(img), np.amax(img)
            koef = 255.0 / (S_max - S_min)
            row = np.multiply(np.add(img, -S_min), koef).astype(np.uint8)
            img = row.reshape(sizes[row_ind])
            cv2.imwrite(os.path.join(path, "base"+str(row_ind)+".pgm"), img)
            
    def __write_pca_images(self, path, sizes) :
        """
        Helper (debug) method for writing the reduced PCA images to a folder.
        """
        for row_ind in range(len(self.__transformed_images)) :
            img = self.__pca.inverse_transform(self.__transformed_images[row_ind])
            S_min, S_max = np.amin(img), np.amax(img)
            koef = 255.0 / (S_max - S_min)
            row = np.multiply(np.add(img, -S_min), koef).astype(np.uint8)
            img = row.reshape(sizes[row_ind])
            cv2.imwrite(os.path.join(path, "pca_trans"+str(row_ind)+".pgm"), img)
            
    def __readFacesOrder(self, train_set_info_path):
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
            
    def __write_reconstructed_images(self, path, sizes) : 
        """
        Helper (debug) method for writing the images reconstructed from learned
        coefficients to a folder.
        """
        for coeff_ind in range(len(self.__original_images_repres)) :
            coeff = self.__original_images_repres[coeff_ind]
            img_rec = self.__pca.inverse_transform(np.dot(coeff, self.__basis_images))
            S_min, S_max = np.amin(img_rec), np.amax(img_rec)
            koef = 255.0 / (S_max - S_min)
            row = np.multiply(np.add(img_rec, -S_min), koef).astype(np.uint8)
            img = row.reshape(sizes[coeff_ind])
            cv2.imwrite(os.path.join(path, "recons"+str(coeff_ind)+".pgm"), img)
            
    def get_similarities_for(self, image_path, similarity_type="cosine") :
        """
        Central method of this class. For a new given image path returns
        a list of triples [(index, similarity, person_index)] 
        which represents the similarity
        between the new image and the images saved/learned.
        
        Similarity can be neative so the rule "the more positive the similarity
        is two images are more similar" applies. similarity_type can be either
        'cosine' or 'euclid'.
        """
        img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
        img = self.__resize_img(img, self.__approximate_dim_num)
        if (self.__equalize_hist) :
            img = cv2.equalizeHist(img)
        flattened = np.array(img.flatten()) / 255.0
        flattened -= self.__pixel_means
        pca_repres = self.__pca.transform(flattened)
        pca_repres -= self.__transformed_images_mean
        transformed = np.dot(self.__transformation, pca_repres[0])
        similarities = []
        for original in self.__original_images_repres :
            simil = 0
            if (similarity_type == "cosine"):
                simil = 1 - cosine(transformed, original);
            elif (similarity_type == "euclid"):
                simil = np.divide(1, 1 + euclidean(transformed, original))
            similarities.append(simil)
        pic_indices = range(0, len(similarities))
        fold_indices = map(lambda x : self.__face_index_lst[x], pic_indices)
        return zip(pic_indices, similarities, fold_indices)
    

def main() :
    BASE_IMG_DIR = './FDDB/train_set'.decode('utf-8')
    rec = RecognitionArh2(BASE_IMG_DIR)
    picture_paths = os.listdir(BASE_IMG_DIR)
    picture_paths = filter(lambda x : \
      os.path.isfile(os.path.join(BASE_IMG_DIR, x)) \
      and unicode.endswith(x, ".pgm"), picture_paths)
    for f in picture_paths:
        path = os.path.join(BASE_IMG_DIR, f)
        simils = rec.get_similarities_for(path, similarity_type="euclid")
        pos, sim = max(simils, key = lambda x : x[1])
        print(str(pos) + " " + str(sim))
    
        
'''
architecture 1 code, might be useful
def create_pixel_obser(pixel_observations, sizes) :
    ica = FastICA(whiten=True)
    S_ = ica.fit_transform(pixel_observations)
    print(ica.components_.shape)
    
    print(ica.mean_)
    
    S_T = S_.T
    print(S_T.shape)
    for row_ind in range(S_T.shape[0]) :
        S_min, S_max = np.amin(S_T[row_ind]), np.amax(S_T[row_ind])
        koef = 255.0 / (S_max - S_min)
        row = np.multiply(np.add(S_T[row_ind], -S_min), koef).astype(np.uint8)
        img = row.reshape(sizes[row_ind])
        cv2.imwrite(os.path.join(BASE_IMG_DIR, "test"+str(row_ind)+".pgm"), img)
        
'''
    
if __name__ == "__main__":
    main()
