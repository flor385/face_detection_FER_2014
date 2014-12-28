import histogram
import util
import cv2
import collections
import numpy
import os
from operator import itemgetter

#metoda koja generira masku
def generate_mask(image, fold):
    ranges = histogram.face_range(fold)
    YIQ_image = util.rgb_to_yiq(image)

    threshold_image = numpy.zeros((len(YIQ_image), len(YIQ_image[0]), 3), numpy.uint8)

    #prodi kroz sve piksele, uzmi one koji zadovoljavaju pragove
    for row in range(0, len(YIQ_image)):
        for col in range(0, len(YIQ_image[0])):
            if YIQ_image[row][col][1] >= ranges[1][0] and YIQ_image[row][col][1] <= ranges[1][1] and YIQ_image[row][col][2] >= ranges[2][0] and YIQ_image[row][col][2] <= ranges[2][1]:
                threshold_image[row][col] = 255
            else:
                threshold_image[row][col] = 0

    #filtriraj masku kako bi se uklonio sum
    threshold_image = cv2.medianBlur(threshold_image,9)
    
    return threshold_image

#metoda koja vraca bounding boxove u formatu [[(x1,y1), (x2,y2)], [(x3,y3), (x4,y4)], ....]
#ukratko, lista unutar koje su liste sa koordinatama gornje lijeve i donje desne tocke
def detect_face_regions(image, fold):
    mask = generate_mask(image, fold)

    pixel_sections = numpy.zeros((len(image), len(image[0])), numpy.uint8)

    #iniciraj susjedstva na prazan dict
    borders = {}

    current_section = 0
    for row in range(0, len(mask)):
        for col in range(0, len(mask[0])):
            #ako je piksel bijel
            if mask[row][col].any():            
                upper_value = 0
                left_value = 0
                #ako postoji nesto odozgo
                if row != 0:
                    upper_value = pixel_sections[row-1][col]
                #ako postoji nesto s lijeva
                if col != 0:
                    left_value = pixel_sections[row][col-1]
                #postavi vrijednost na susjeda. Manji susjed ima prioritet
                if left_value != 0 or upper_value != 0:
                    if left_value == 0:
                        pixel_sections[row][col] = upper_value
                    elif upper_value == 0:
                        pixel_sections[row][col] = left_value
                    else:
                        pixel_sections[row][col] = min(upper_value, left_value)
                        #dodaj susjedstvo ako susjedi nisu nule te ako su razliciti
                        if upper_value != left_value:
                            borders[max(left_value,upper_value)] = min(left_value,upper_value)
                if pixel_sections[row][col] == 0:
                    #print 'povecano ' + str(current_section)
                    current_section += 1
                    pixel_sections[row][col] = current_section
    
    #susjedstva sortiraj
    borders = collections.OrderedDict(sorted(borders.items()))

    
    #pojednostavi susjedstva
    for key in borders:
        if borders[key] in borders:   
            borders[key] = borders[borders[key]]

    #ovisno o susjedstvima promjeni vrijednosti piksela    
    for row in range(0, len(pixel_sections)):
        for col in range(0, len(pixel_sections[0])):
            if pixel_sections[row][col] in borders:
                pixel_sections[row][col] = borders[pixel_sections[row][col]]

    #uzmi moguce vrijednosti piksela, izbacujuci crne piksele
    possible_values = sorted(set(i for j in pixel_sections for i in j))
    possible_values.pop(0)

    #podesi dict koji ce svaku mogucu vrijednost segmenta piksela popisati sve pripadne piksele
    value_mapping = {}
    for value in possible_values:
        value_mapping[value] = []

    #pridodaj svakoj vrijednosti njene piksele
    for row in range(0, len(pixel_sections)):
        for col in range(0, len(pixel_sections[0])):
            if pixel_sections[row][col] != 0:
                value_mapping[pixel_sections[row][col]].append([row, col])

    images = []

    #prodi kroz sve vrijednosti te za svaku napravi kvadrat koji opisuje njene piksele
    #spremi sve pronadene kvadrate kao slike
    for key in value_mapping:
        max_x = max(value_mapping[key], key=itemgetter(0))[0]
        max_y = max(value_mapping[key], key=itemgetter(1))[1]
        min_x = min(value_mapping[key], key=itemgetter(0))[0]
        min_y = min(value_mapping[key], key=itemgetter(1))[1]

        if max_x - min_x > 10 and max_y - min_y > 10:
            images.append(([min_y, min_x], [max_y, max_x]))
            """
            cv2.rectangle(image, (min_y, min_x), (max_y, max_x), (0,0,0), 2)
            

    
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """        
    #vrati bounding boxove       
    return images

#metoda koja testira cijelu stvar
def test():
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_115.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_257.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_276.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_362.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_475.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_496.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_498.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_558.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_591.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_643.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_674.jpg'), 1)
    detect_face_regions(image, 1)
    image = cv2.imread(os.path.join('FDDB', '2002', '08', '11', 'big', 'img_752.jpg'), 1)
    detect_face_regions(image, 1)
