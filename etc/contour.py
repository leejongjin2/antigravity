import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

def save_mask_seperately(maskpath):
    os.makedirs(maskpath + "/sep", exist_ok=True)
    for mask_image in glob.glob(maskpath + "*_mask.png"):
        mask = cv2.imread(mask_image)
        mask_ = cv2.cvtColor(mask.copy(), cv2.COLOR_BGR2GRAY)
        contours, hierarchy = cv2.findContours(mask_, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        img = np.zeros_like(mask)
        for i in range(len(contours)):
            contour_mask = cv2.drawContours(img.copy(), [contours[i]], -1, color=(100, 100, 100), thickness=-1)
            original_file_name = "sep/" + os.path.basename(mask_image).split(".")[0] + f"_{i}.png"
            original_file_name = os.path.join(maskpath, original_file_name)
            cv2.imwrite(original_file_name, contour_mask)
    
def file_path(*args):
    path = ""
    for arg in args:
        path = os.path.join(path, arg)

    return path


if __name__=="__main__":
    # maskpath = "C:/Users/antigravity/workspace/dataset_/inimages/seg/"
    # save_mask_seperately(maskpath)

    data_path = "C:/Users/antigravity/workspace/dataset_/coupang/rgbd_ak4"
    mask_suffix = "_mask"
    rgb_prefix = "rgb"
    depth_prefix = "depth"
    image_extension = ".png"
    depth_extension = ".npy"
    image_number = (0, 28)
    k = 386

    os.makedirs(os.path.join(data_path, "sep"), exist_ok=True)
    for i in tqdm(range(image_number[0], image_number[1] + 1)):
        rgb_file_name = f"{rgb_prefix}{i}"
        mask_file_name = f"{rgb_file_name}{mask_suffix}"
        mask_path = file_path(data_path, f"{mask_file_name}{image_extension}")
        rgb_path = file_path(data_path, f"{rgb_file_name}{image_extension}")
        depth_path = file_path(data_path, f"{depth_prefix}{i}{depth_extension}")

        mask = cv2.imread(mask_path, 0)
        rgb = cv2.imread(rgb_path)
        depth = np.load(depth_path)

        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        img = np.zeros_like(mask)

        for i in range(len(contours)):
            contour_mask = cv2.drawContours(img.copy(), [contours[i]], -1, color=(100, 100, 100), thickness=-1)
            
            masked_rgb = rgb.copy()
            masked_rgb[contour_mask == 0] = 0

            masked_depth = depth.copy()
            masked_depth[contour_mask == 0] = 0

        
            bbox = list(zip(map(min, np.where(contour_mask)), map(max, np.where(contour_mask))))
            cropped_rgb = masked_rgb[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]
            cropped_depth = masked_depth[bbox[0][0]:bbox[0][1], bbox[1][0]:bbox[1][1]]

            cv2.imwrite(f"{data_path}/sep/rgb{k}{image_extension}", cropped_rgb)
            np.save(f"{data_path}/sep/rgb{k}{depth_extension}", cropped_depth)
            k += 1
