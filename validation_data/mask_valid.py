import os
import numpy as np
import cv2
import json
import argparse
from argparse import Namespace
import glob
from pathlib import Path
import sys
import time

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def data_read(root_path:str):
    data_lst = []
    landmark_colors = []
    pad_colors = []
    wire_colors = []
    file_lst = glob.glob(os.path.join(root_path,'images/*'))

    for file in file_lst:
        maskname = os.path.basename(file).replace('img', 'mask')
        mask_file  = os.path.join(root_path, 'segmentations', maskname)
        if os.path.isfile(mask_file):
            print('OK')
        else:
            print('sdafjklasf')
        data_lst.append([file, mask_file])

    with open(args.config, 'r') as f:
        colors = json.load(f)
    for k, v in colors.items():
        if 'landmark' in k:
            landmark_colors.append(tuple(v))
        elif 'pad' in k:
            pad_colors.append(tuple(v))
        elif 'wire' in k:
            wire_colors.append(tuple(v))

    return data_lst, landmark_colors, pad_colors, wire_colors

def imshow(winname:str, img:np.ndarray, width:int=1500, height:int=1200) -> None:
    cv2.namedWindow(winname, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname=winname, width=width, height=height)
    cv2.imshow(winname, img)

def get_landmark(mask: np.ndarray, colors: list) -> list:
    landmark_lst = []
    for i, color in enumerate(colors):
        mask_copy = mask.copy()
        mask_copy = np.array((mask_copy==color)*255, dtype=np.uint8)
        gray = cv2.cvtColor(mask_copy, cv2.COLOR_BGR2GRAY)
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1 = 1, param2 = 1, minRadius = 0, maxRadius = 50)
        landmark_lst += [[int(circles[0][0][0]), int(circles[0][0][1])]]
    return landmark_lst

def get_segmask(mask: np.ndarray, colors) -> np.ndarray:
    h, w, c = mask.shape
    pad_mask = np.zeros((h, w, c), dtype=np.uint8)
    for color in colors:
        mask_copy = mask.copy()
        mask_copy = np.array((mask_copy==color)*255, dtype=np.uint8)
        pad_mask += mask_copy

    return pad_mask

def onMouse(x):
    pass

def img_blending(path, img, pad, wire, landmarks):
    cv2.namedWindow('imgPane')
    cv2.createTrackbar('PAD', 'imgPane', 50, 100, onMouse)
    cv2.createTrackbar('WIRE', 'imgPane', 25, 100, onMouse)

    img_raw = img.copy()
    img_circle = img.copy()
    for landmark in landmarks:
        cv2.circle(img_circle, (landmark), 3, (0, 255, 0), -1)
    while True:
        pad_mix = cv2.getTrackbarPos('PAD', 'imgPane')
        wire_mix = cv2.getTrackbarPos('WIRE', 'imgPane')
        result = cv2.addWeighted(img, float(100-pad_mix)/100, pad, float(pad_mix)/100, 0)
        result = cv2.addWeighted(result, float(100-wire_mix)/100, wire, float(wire_mix)/100, 0)
        k = cv2.waitKey(1) & 0xFF
        if k == 27: # ESC : Next Image
            break
        elif k == ord('i') or k == ord('I'): # I : Draw Circle Image
            img = img_circle
        elif k == ord('w') or k == ord('w'): # W : Raw Image (Original)
            img = img_raw
        elif k == ord('q') or k == ord('Q'): # Q : Exit
            print(path)
            sys.exit()
        imshow('imgPane', result)
        pad_mix = cv2.getTrackbarPos('PAD', 'imgPane')
        wire_mix = cv2.getTrackbarPos('WIRE', 'imgPane')



def main(args):
    data_lst, landmark_colors, pad_colors, wire_colors = data_read(args.root_path)

    for img_path, mask_path in data_lst:
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        pad_mask = get_segmask(mask=mask, colors=pad_colors) # Get Pad Mask (Image)
        wire_mask = get_segmask(mask=mask, colors=wire_colors) # Get Wire Mask (Image)
        landmarks = get_landmark(mask, colors=landmark_colors) # Get Landmark Point (list[x,y])

        img_blending(img_path, img, pad_mask, wire_mask, landmarks)
        cv2.destroyAllWindows()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default= ROOT / 'mask_config.json', help='model.pt path')
    parser.add_argument('--root_path', type=str, default='/home/ljj/dataset/chec', help='model.pt path')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)