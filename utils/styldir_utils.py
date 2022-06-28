import torch

import numpy as np
from PIL import Image

import cv2

def network_output2image(image_batches):
    image_batches = ((image_batches.permute(0, 2, 3, 1).clip(-1, 1).detach().cpu().numpy() + 1) * 127.5).astype('uint8')
    return image_batches
def display_image_group(images, size = 256):
    res = np.array(images[0].resize((size, size)))
    for i in range(1, len(images)):
        res = np.concatenate([res, images[i].resize((size, size))], axis = 1)


    return Image.fromarray(res)

def display_landmarks(landmarks, only_lips = True, size = 256):
    image = (np.ones((size, size, 3)) * 255).astype('uint8')
    if only_lips:
        landmarks = landmarks[48:]
    for l in landmarks:
        image = cv2.circle(image, center=tuple(l), radius=2, color=(0, 0, 255), thickness=-100)

    return image

def mask_image(image, landmarks, draw_on_image = True, inplace = False):
    mask_points = landmarks[2:15]
    image_dims = image.shape

    if not inplace:
        image = image.copy()
    if not draw_on_image:
        mask = np.zeros(image_dims)
        mask = cv2.fillPoly(mask, [mask_points], color=(255, 255, 255)).astype('uint8')
        mask = mask[:, :, 0].astype('float')
        mask[mask == 255] = 1.
    else:
        mask = cv2.fillPoly(image, [mask_points], color=(255, 255, 255)).astype('uint8')

    return mask

def draw_landmarks_on_image(img, landmarks):
    img = img.copy()
    for l in landmarks:
        img = cv2.circle(img=img, center=tuple(l), radius=1, color=(0, 0, 255), thickness=-10)

    return img