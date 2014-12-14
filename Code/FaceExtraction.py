import numpy
import matplotlib
import cv2
import math
import os
from os import listdir
from os.path import isfile, join

IMAGE_SIZE = 32

#pomocna metoda koja validira poziciju piksela
def legitimize(number):
    if number < 0:
        number = 0;
    return int(number)

def check_if_in_ellipse(point_x, point_y, major_axis, minor_axis, angle, center_x, center_y):
    cosa=math.cos(angle)
    sina=math.sin(angle)
    dd=minor_axis*minor_axis
    DD=major_axis*major_axis

    a =math.pow(cosa*(point_x-center_x)+sina*(point_y-center_y),2)
    b =math.pow(sina*(point_x-center_x)-cosa*(point_y-center_y),2)
    ellipse=(a/DD)+(b/dd)

    if ellipse <= 1:
        return True
    else:
        return False

#metoda koja iz slike izrezuje lice opisano elipsom
def crop_image(image, ellipse_info):
    backup = image
    image = cv2.imread(image, 1)
    info_array = ellipse_info.split(" ")
    major_axis = int(float(info_array[0]))
    minor_axis = int(float(info_array[1]))
    angle = float(info_array[2])
    center_x = int(float(info_array[3]))
    center_y = int(float(info_array[4]))

    #puno matematike, stack overflow link: http://stackoverflow.com/questions/87734/how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse
    ux = major_axis * math.cos(angle);
    uy = major_axis * math.sin(angle);
    vx = minor_axis * math.cos(angle+math.pi/2);
    vy = minor_axis * math.sin(angle+math.pi/2);

    bbox_halfwidth = math.sqrt(ux*ux + vx*vx);
    bbox_halfheight = math.sqrt(uy*uy + vy*vy); 

    bbox_ul_corner = (legitimize(center_x - bbox_halfwidth), legitimize(center_y - bbox_halfheight))
    bbox_br_corner = (legitimize(center_x + bbox_halfwidth), legitimize(center_y + bbox_halfheight))


    crop_x = min(bbox_ul_corner[0], bbox_br_corner[0])
    crop_y = min(bbox_ul_corner[1], bbox_br_corner[1])

    width = max(bbox_ul_corner[0], bbox_br_corner[0]) - min(bbox_ul_corner[0], bbox_br_corner[0])
    height = max(bbox_ul_corner[1], bbox_br_corner[1]) - min(bbox_ul_corner[1], bbox_br_corner[1])

    for i in range(crop_y, crop_y + height):
        for j in range(crop_x, crop_x + width):
            if(not check_if_in_ellipse(j, i, major_axis, minor_axis, angle, center_x, center_y)):
               for k in range(0,3):
                   try:
                        image[i][j][k] = 0
                   except IndexError as detail:
                        #print crop_y + i, crop_x + j
                        message = "Ne znam kako je ovo uopce moguce, iduca linija koristi maksimalne vrijednosti"
                            
    return image[crop_y:crop_y + height, crop_x:crop_x+width]

#metoda koja pretvara RGB sliku u YIQ sliku
def convert_RGB_to_YIQ(img):
    YIQ_image = numpy.zeros((len(img), len(img[0]), 3), numpy.uint8)
    YIQ_current = numpy.arange(3).reshape((3,1))
    YIQ_weights = numpy.matrix("0.299 0.587 0.114; 0.596 -0.275 -0.320; 0.212 -0.523 0.311")

    RGB_current = numpy.arange(3).reshape((3,1))

    for row in range(0, len(img)):
        for col in range(0, len(img[row])):
            RGB_current[0][0] = img[row][col][0]
            RGB_current[1][0] = img[row][col][1]
            RGB_current[2][0] = img[row][col][2]
            YIQ_current = numpy.dot(YIQ_weights, RGB_current)
            YIQ_image[row][col][0] = int(YIQ_current[0][0])
            YIQ_image[row][col][1] = int(YIQ_current[1][0])
            YIQ_image[row][col][2] = int(YIQ_current[2][0])

            if RGB_current[0][0] == 0 and RGB_current[0][0] == 0 and RGB_current[0][0] == 0:
                YIQ_image[row][col][0] = 255
                YIQ_image[row][col][1] = 255
                YIQ_image[row][col][2] = 255

    return YIQ_image

#metoda koja iz danih podataka nalazi sve slike, ucitava ih te sprema nadena lica
#pri citanju podataka ce se citati sve datoteke u pathu, tako da se treba staviti samo one koje se zeli koristiti
#koriste se samo .txt datoteke sa kompletnim anotacijama za elipse, one bez toga nece proci
def generate_faces(image_path, information_path, extraction_path):
    current_face = 1

    image_information = [ f for f in listdir(information_path) if isfile(join(information_path,f)) ]
    current_image = ""

    for image_information_file in image_information:
        current_file = open(information_path + os.sep + image_information_file)
        lines = current_file.readlines()
        current_file.close()
        flag = "path_line"
        repeat = 0;
        for line in lines:
            line = line.strip()
            if flag == "path_line":
                flag = "number_line"
                current_image = image_path + os.sep + line + ".jpg"
                continue
            if flag == "number_line":
                flag = "ellipse_line"
                repeat = int(line)
                continue;
            if flag == "ellipse_line":
                face_image = crop_image(current_image, line)
                #koristi samo slike koje su dovoljno velike pa ih resizeaj
                if len(face_image) >= 32 and len(face_image[0]) >= 32:
                    face_image = cv2.resize(face_image, (IMAGE_SIZE,IMAGE_SIZE))
                    cv2.imwrite(extraction_path + os.sep + "face" + str(current_face) + ".jpg", face_image)
                current_face += 1
                repeat -= 1
                if repeat == 0:
                    flag = "path_line"

#metoda koja generira prosjecno lice iz danih lica
def generate_average_face(faces_path, average_face_path):
    face_images = [ f for f in listdir(faces_path) if isfile(join(faces_path,f)) ]

    average_face= [[[0 for col in range(3)]for row in range(IMAGE_SIZE)] for x in range(IMAGE_SIZE)]

    number_of_faces = len(face_images)
    for face_image in face_images:
        face = cv2.imread(faces_path + os.sep + face_image, 1)
        for row in range(0, len(face)):
            for col in range(0, len(face[0])):
                average_face[row][col][0] += float(face[row][col][0]) / float(number_of_faces)
                average_face[row][col][1] += float(face[row][col][1]) / float(number_of_faces)
                average_face[row][col][2] += float(face[row][col][2]) / float(number_of_faces)

    for row in range(0, len(average_face)):
        for col in range(0, len(average_face[0])):
            average_face[row][col][0] = int(average_face[row][col][0])
            average_face[row][col][1] = int(average_face[row][col][1])
            average_face[row][col][2] = int(average_face[row][col][2])

    cv2.imwrite(average_face_path + os.sep + "average_face.jpg", numpy.asarray(average_face))
    cv2.imwrite(average_face_path + os.sep + "average_face_YIQ.jpg", convert_RGB_to_YIQ(numpy.asarray(average_face)))


