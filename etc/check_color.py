import numpy as np
import cv2
import glob
from pathlib import Path

x1,y1 = -1,-1

def find_pixel(event, x, y, flags, param):
    global x1,y1, mask

    if event == cv2.EVENT_LBUTTONDOWN:                      # 마우스를 누른 상태
        x1, y1 = x,y
        print(x1, y1)
        print(mask[y1,x1])

        # cv2.circle(window, (x1, y1), 5, (255, 0, 255), -1)
        cv2.imshow("image", mask)

path = '/home/ljj/dataset/anti_sample'
file_lst = glob.glob(path+"/*")

img_lst = []
mask_lst = []


for file in file_lst:
    if "_img" in file:
        p = Path(file)
        mask_file  = str(p.parent / (p.stem[:-3] + 'mask.png'))
        img_lst.append(file)
        mask_lst.append(mask_file)

mask = cv2.imread(mask_lst[0])

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow(winname='image', width=1500, height=1000)
cv2.imshow("image", mask)
cv2.setMouseCallback('image', find_pixel)

while True:
    cv2.imshow('image', mask)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
