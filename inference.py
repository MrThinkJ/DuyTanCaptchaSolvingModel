import os.path
import torch.nn as nn
import numpy as np
import torch
import cv2
from image_processing import extract_character


def image_processing(images_path, index):
    image_path = os.path.join(images_path, "{}.png".format(index))
    ori_image = cv2.imread(image_path)
    image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (30, 30))
    image = np.expand_dims(image, 2)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, 0)
    return torch.from_numpy(image).float()


def inference(data_path, model, categories):
    images_path = "data_inference"
    softmax = nn.Softmax(dim=1)
    model.eval()
    extract_character(data_path)
    result = ""
    for j in range(4):
        image = image_processing(images_path, j)
        with torch.no_grad():
            prediction = model(image)
            prob = softmax(prediction)
        _, max_index = torch.max(prob, dim=1)
        result += categories[max_index[0]]
    return result
