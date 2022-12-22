import glob
import cv2
import numpy as np
import os
from mmdet.apis import inference_detector, init_detector
from pathlib import Path
from typing import Tuple, Union
import argparse
from utils import get_result_info, get_bbox_and_mask
import time


def show_img(window_name, img, width=1500, height=1000):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(winname=window_name, width=width, height=height)
    cv2.imshow(window_name, img)

def main(args):
    maskrcnn_model = init_detector( args.config, args.model_path, device="cuda")
    maskrcnn_model = maskrcnn_model.eval()
    for img_path in glob.glob(args.input_path+'*'):

    # img_path = args.input_path

        img_name = Path(img_path).parts[-1]
        if 'mask' not in img_name:
            img = cv2.imread(img_path)
            st = time.time()
            result = inference_detector(maskrcnn_model, img)
            # Get Mask
            mask_img, bbox_int, mask = get_result_info(img, result)
            # Crop
            cropped_img, mask = get_bbox_and_mask(img, bbox_int, mask, offset = 10)
            print(time.time() - st)


            show_img('img', img)
            show_img('mask', mask_img)
            show_img('cropped_img', cropped_img)
            cv2.waitKey(0)

        # cv2.imwrite(os.path.join(args.result_path, img_name), img)


def parse_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--config', help='train config file path')
    parser.add_argument('--config', type=str, default='/home/ljj/workspace/antigravity/snuailab_dev/work_dirs/pad_swin/pad_swin_v1_config.py', help='config path')
    parser.add_argument('--model_path', type=str, default='/home/ljj/workspace/antigravity/snuailab_dev/work_dirs/pad_swin/epoch_23.pth', help='model.pt path')
    parser.add_argument('--result_path', type=str, default='/home/ljj/workspace/antigravity/test/result', help='result path')
    parser.add_argument('--input_path', type=str, default='/home/ljj/workspace/antigravity/snuailab_dev/data/pad_sample/', help='result path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.9, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show', action='store_true', help='display tracking video results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)