import numpy as np
import cv2
import glob
import os
x1, y1 = -1, -1
def find_pixel(event, x, y, flags, param):
    global x1, y1, window
    if event == cv2.EVENT_LBUTTONDOWN:
        x1 = x
        y1 = y
        print(x1, y1)
        print(window[y1, x1])
        # cv2.circle(window, (x1, y1), 5, (255, 0, 255), -1)
        cv2.imshow('image', window)

path = '/home/ljj/dataset/chec'
seg_path = os.path.join(path, 'segmentations/*')
for data_path in glob.glob(seg_path):
    print(data_path)
    img = cv2.imread(data_path)
    window = img.copy()

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname='image', width=1500, height=1000)
    cv2.imshow("image", window)
    cv2.setMouseCallback('image', find_pixel)

    while True:
        cv2.imshow('image', window)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

