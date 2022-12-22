import glob
from utils import create_image_annotation, get_pad, get_landmark, create_annotation_format,\
     create_sub_mask_annotation, MultiPolygon, get_coco_json_format, create_category_annotation
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image
import os
import json
from arguments import parse_args


category_ids = {
    "pad1": 0,
    "pad2": 1,
}
category_colors = {
    "(100, 100, 100)": 0, # pad Left
    "(101, 101, 101)": 1, # pad Right
}
def images_annotations_info(args):
    landmark_colors = []
    multipolygon_ids = []
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []

    with open(args.config, 'r') as f:
        config = json.load(f)
    for k, v in config.items():
        if 'landmark' in k and 'c' not in k:
            landmark_colors.append(tuple(v))

    for mask_image in tqdm(glob.glob(os.path.join(args.root_path, "segmentations/*"))):
        # original_file_name = os.path.basename(mask_image).split(".")[0].replace("_mask", "") + "_img.png"
        image_filename = os.path.basename(mask_image).replace('mask','img')
        mask_cv2=cv2.imread(mask_image)
        h, w = mask_cv2.shape[:2]
        
        image = create_image_annotation(image_filename, w, h, image_id)
        images.append(image)

        sub_mask1 = get_pad(mask_cv2, colors=[(33, 33, 33), (11, 11, 11), (5,5,5), (26,26,26), (29,29,29)])
        key_points1 = get_landmark(mask_cv2, colors = [(5,5,5), (26,26,26), (29,29,29)])
        sub_mask2 = get_pad(mask_cv2, colors=[(15, 15, 15), (28, 28, 28), (6,6,6), (13,13,13), (1,1,1)])
        key_points2 = get_landmark(mask_cv2, colors = [(6,6,6), (13,13,13), (1,1,1)])
        gray1 = cv2.cvtColor(sub_mask1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(sub_mask2, cv2.COLOR_BGR2GRAY)
        sub_masks = {'(100, 100, 100)': [gray1, key_points1], '(101, 101, 101)': [gray2, key_points2]}

        for color, (sub_mask, key_points) in sub_masks.items():
            polygons, segmentations = create_sub_mask_annotation(sub_mask)
            category_id = category_colors[color]

            if category_id in multipolygon_ids:
                multi_poly = MultiPolygon(polygons)
                annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id, keypoint=key_points)
                annotations.append(annotation)
                annotation_id += 1
            else:
                for i in range(len(polygons)):
                    segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                    annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id, keypoint=key_points)
                    annotations.append(annotation)
                    annotation_id += 1
        image_id += 1
    return images, annotations, annotation_id


if __name__ == "__main__":
    args = parse_args()

    coco_format = get_coco_json_format()
    coco_format["categories"] = create_category_annotation(category_ids)
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(args)

    with open(os.path.join(args.root_path, "pad.json"),"w") as outfile:
        json.dump(coco_format, outfile)
