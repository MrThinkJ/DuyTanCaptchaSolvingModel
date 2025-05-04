import cv2
import numpy
import numpy as np
import os


def median_filter_h(image, kernel):
    h = image.shape[0]
    w = image.shape[1]
    z = kernel // 2
    data_final = numpy.zeros((h, w))
    for i in range(h):
        for j in range(w):
            temp = []
            for k in range(kernel):
                if i + k < z or i + k - z > h - 1:
                    temp.append(0)
                else:
                    temp.append(image[i + k - z][j])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
    return data_final


def median_filter_w(image, kernel):
    h = image.shape[0]
    w = image.shape[1]
    z = kernel // 2
    data_final = numpy.zeros((h, w))
    for i in range(h):
        for j in range(w):
            temp = []
            for k in range(kernel):
                if j + k < z or j + k - z > w - 1:
                    temp.append(0)
                else:
                    temp.append(image[i][j + k - z])
            temp.sort()
            data_final[i][j] = temp[len(temp) // 2]
    return data_final


def cut(image, w, h):
    return image[w:-w, h:-h]


def extract_character(input_image, output_folder="data_inference"):
    img = cv2.imread(input_image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.bitwise_not(thresh1)
    img_erosion = cv2.erode(image, np.ones((2, 2), dtype=np.uint8), iterations=1)
    img_filter = median_filter_w(img_erosion, 2)

    offset_h = 4
    bh = 0
    eh = img_gray.shape[0]
    part1 = img_filter[bh + offset_h: eh - offset_h, 5:30]
    part2 = img_filter[bh + offset_h: eh - offset_h, 30:55]
    part3 = img_filter[bh + offset_h: eh - offset_h, 60:85]
    part4 = img_filter[bh + offset_h: eh - offset_h, 90:115]

    parts = [part1, part2, part3, part4]

    for index, part in enumerate(parts):
        save_path = os.path.join(output_folder)
        p = os.path.join(save_path, "{}.png".format(index))
        cv2.imwrite(p, part)
