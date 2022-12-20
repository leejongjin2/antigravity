import numpy as np
import cv2
import glob
from pathlib import Path
import json





def get_pad(mask: np.ndarray, colors) -> np.ndarray:
    h, w, c = mask.shape
    pad_mask = np.zeros((h, w, c), dtype=np.uint8)
    for color in colors:
        mask_copy = mask.copy()
        mask_copy = np.array((mask_copy==color)*255, dtype=np.uint8)
        pad_mask += mask_copy

    return pad_mask

def get_landmark(mask: np.ndarray, colors: list) -> list:
    landmark_lst = []
    for i, color in enumerate(colors):
        mask_copy = mask.copy()
        mask_copy = np.array((mask_copy==color)*255, dtype=np.uint8)
        # cv2.imshow('mask', mask_copy)
        # cv2.waitKey(0)
        gray = cv2.cvtColor(mask_copy, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 1, param2 = 1, minRadius = 0, maxRadius = 50)
        # for i in circles[0]:
        #     landmark_lst.append([int(i[0]), int(i[1])])
    
        landmark_lst += [int(circles[0][0][0]), int(circles[0][0][1]), 1]
    return landmark_lst


if __name__ == 'main':    
    path = '/home/ljj/dataset/anti_sample'
    file_lst = glob.glob(path+"/*")

    img_lst = []
    mask_lst = []

    # file path
    for file in file_lst:
        if "_img" in file:
            p = Path(file) # img
            mask_file  = str(p.parent / (p.stem[:-3] + 'mask.png')) # mask
            img_lst.append(file)
            mask_lst.append(mask_file)

    #### json
    json_path = '/home/ljj/workspace/antigravity/etc/mask_config.json'
    with open(json_path, 'r') as f:
        config = json.load(f)

    pad_colors = []
    landmark_colors = []
    for k, v in config.items():
        if 'landmark' in k:
            landmark_colors.append(tuple(v))
        elif 'pad' in k:
            pad_colors.append(tuple(v))

    img = cv2.imread(img_lst[0])
    mask = cv2.imread(mask_lst[0])

    pad_mask = get_pad(mask=mask, colors=pad_colors)
    landmarks = landmark_det(mask=mask, colors=landmark_colors)
    for landmark in landmarks:
        cv2.circle(img, (landmark[0], landmark[1]), 4, (255,0,255), -1)


    print(landmarks)
    cv2.imshow('img', img)
    cv2.imshow('mask', pad_mask)
    cv2.waitKey(0)
