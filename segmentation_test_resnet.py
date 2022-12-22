import glob
import cv2
from mmdet.apis import inference_detector, init_detector
from pathlib import Path
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
        img_name = Path(img_path).parts[-1]
        img = cv2.imread(img_path)
        st = time.time()
        result = inference_detector(maskrcnn_model, img)
        # Get Mask
        mask_img, bbox_int, mask = get_result_info(img, result)
        # Crop
        cropped_img, mask = get_bbox_and_mask(img, bbox_int, mask, args.offset)
        print(time.time() - st)


        show_img('img', img)
        show_img('mask', mask_img)
        show_img('cropped_img', cropped_img)
        cv2.waitKey(0)

        # cv2.imwrite(os.path.join(args.result_path, img_name), img)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/home/ljj/models/mask_rcnn_pad_config.py', help='config path')
    parser.add_argument('--model_path', type=str, default='/home/ljj/models/latest.pth', help='model.pt path')
    parser.add_argument('--result_path', type=str, default='/home/ljj/workspace/antigravity/test/result', help='result path')
    parser.add_argument('--input_path', type=str, default='/home/ljj/dataset/anti102/images/', help='result path')
    parser.add_argument('--offset', type=int, default=10, help='confidence threshold')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)