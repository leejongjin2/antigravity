import cv2
import numpy as np
import os
from typing import Tuple, Union

def get_result_info(img: np.ndarray, result: tuple) -> Tuple[np.ndarray, list, np.ndarray]:
    threshold=0.0
    img = img.copy()
    bboxes, mask = result
    msk, bbox_int = None, None
    for i, bbox in enumerate(bboxes[0]):
        if bbox[-1] < threshold: # Confidence Score에 대한 threshold가 넘지 못하면 box가 아니라고 판단.
            continue
        score = bbox[-1]
        bbox_int = bbox.astype(np.int32)
        msk = mask[0][i]
        mask_img = img.copy()
        mask_img[msk] = [191, 255, 0]

        img = cv2.addWeighted(img, 0.7, mask_img, 0.3, 0)
        img = cv2.rectangle(img, (bbox_int[0], bbox_int[1]), (bbox_int[2], bbox_int[3]), (0, 255, 0), 3)

    return img, bbox_int, msk

def get_bbox_and_mask(
    frame: np.ndarray,
    bbox_int: list,
    mask: np.ndarray,
    offset : int,
) -> Tuple[np.ndarray, list, np.ndarray]:

    if bbox_int is None:
        print("Warning: bbox has not detected.")
        return frame, None, frame

    bbox_size = (
        (bbox_int[3] - bbox_int[1]) / frame.shape[0],
        (bbox_int[2] - bbox_int[0]) / frame.shape[1],
    )

    def cropper(array: np.ndarray, bbox_int: list) -> np.ndarray:
        array = array[bbox_int[1] - offset : bbox_int[3] + offset, bbox_int[0] - offset : bbox_int[2] + offset]
        return array

    img = frame.copy()
    img = cropper(img, bbox_int)
    mask = cropper(mask, bbox_int)

    img[mask == False] = 0
    img = cv2.resize(img, (320, 320))

    return img, mask

