from utils.landmarks_utils import vis_landmark_on_img
from utils.files_utils import alphanum_key
import numpy as np
import cv2
import os
import joblib
import matplotlib.pyplot as plt


target_landmark_dir = "/home/npatel/Documents/lip2lip/target_landmarks.npy"
output_landmark_dir = "/home/npatel/Documents/lip2lip/inference_landmarks.npy"
transform_path = "/home/npatel/Documents/lip2lip_data/target_transform"
target_norm_path = "/home/npatel/Documents/lip2lip_data/normalized_target_landmarks"
paths = sorted([os.path.join(transform_path, file) for file in os.listdir(transform_path)], key=alphanum_key)
transforms = [joblib.load(file) for file in paths]

target_lms = np.load(target_landmark_dir)
output_lms = np.load(output_landmark_dir)
target_lm_paths = sorted([os.path.join(target_norm_path, file) for file in os.listdir(target_norm_path)], key=alphanum_key)

height, width, channels = 3840, 2160, 3
blank_img = np.zeros((height, width * 2, channels), np.uint8)
blank_img[:, :, :] = 255

for i, (t_lms, o_lms) in enumerate(zip(target_lms, output_lms)):
    if i > 20:
        break
    print("Computing image:", i)
    transform = transforms[i]
    landmarks = np.load(target_lm_paths[i])
    landmarks = transform.inverse_transform(landmarks)
    o_lms = transform.inverse_transform(o_lms)
    t_lms = transform.inverse_transform(t_lms)

    landmarks[48:68] = o_lms
    img_out = vis_landmark_on_img(blank_img, landmarks)
    landmarks[48:68] = t_lms
    img_trg = vis_landmark_on_img(blank_img, landmarks)
    side_imgs = np.concatenate([img_trg, img_out], axis=1)

    fig = plt.figure()
    plt.imshow(side_imgs)
    # plt.axis("off")
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig(f'/home/npatel/Documents/lip2lip_data/output compare 2/landmarks_out{i}.png')
    plt.close("all")


root_folder = '/home/npatel/Documents/lip2lip_data/output compare 2'
figures = sorted([os.path.join(root_folder, file) for file in os.listdir(root_folder)], key=alphanum_key)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter('compare_landmarks.mp4', fourcc, 30, (height, width * 2))

for fig in figures:
    img = np.array(cv2.imread(fig))
    writer.write(img)