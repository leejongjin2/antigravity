import glob
import re
from create_annotations import *
from tqdm import tqdm
from contour import save_mask_seperately
import cv2
from landmark_ import get_landmark, get_pad
from PIL import Image

# Label ids of the dataset
category_ids = {
    "pad1": 0,
    "pad2": 1,
}

# Define which colors match which categories in the images


json_path = '/Users/kaejong/workspace/antigravity/etc/mask_config.json'
with open(json_path, 'r') as f:
    config = json.load(f)

# category_colors = {}
# for i, (k, v) in enumerate(config.items()):
#     category_colors[str(tuple(v))] = i
    
category_colors = {
    "(100, 100, 100)": 0, # pad Left
    "(101, 101, 101)": 1, # pad Right
}

landmark_colors = []
for k, v in config.items():
    if 'landmark' in k and 'c' not in k:
        landmark_colors.append(tuple(v))
# Define the ids that are a multiplolygon. In our case: wall, roof and sky
multipolygon_ids = []

# Get "images" and "annotations" info 
def images_annotations_info(maskpath):
    # This id will be automatically increased as we go
    annotation_id = 0
    image_id = 0
    annotations = []
    images = []
    for mask_image in tqdm(glob.glob(maskpath + "*_mask*.png")):
        if 'watershed' not in mask_image and 'color' not in mask_image:
        # The mask image is *.png but the original image is *.jpg.
            # We make a reference to the original file in the COCO JSON file
            original_file_name = os.path.basename(mask_image).split(".")[0].replace("_mask", "") + "_img.png"
            # original_file_name = re.sub("_mask*", "", os.path.basename(mask_image).split(".")[0]) + ".png"
            # mask_image.split("_")[1].split("\\")[-1]

            # Open the image and (to be sure) we convert it to RGB
            # mask_image_open = Image.open(mask_image).convert("RGB")
            # w, h = mask_image_open.size
            mask_cv2=cv2.imread(mask_image)
            h, w = mask_cv2.shape[:2]
            
            # "images" info 
            image = create_image_annotation(original_file_name, w, h, image_id)
            images.append(image)

            # sub_masks = create_sub_masks(mask_image_open, w, h)
            sub_mask1 = get_pad(mask_cv2, colors=[(33, 33, 33), (11, 11, 11), (5,5,5), (26,26,26), (29,29,29)])
            key_points1 = get_landmark(mask_cv2, colors = [(5,5,5), (26,26,26), (29,29,29)])
            sub_mask2 = get_pad(mask_cv2, colors=[(15, 15, 15), (28, 28, 28), (6,6,6), (13,13,13), (1,1,1)])
            key_points2 = get_landmark(mask_cv2, colors = [(6,6,6), (13,13,13), (1,1,1)])
            gray1 = cv2.cvtColor(sub_mask1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(sub_mask2, cv2.COLOR_BGR2GRAY)
            sub_masks = {'(100, 100, 100)': [gray1, key_points1], '(101, 101, 101)': [gray2, key_points2]}

            for color, (sub_mask, key_points) in sub_masks.items():
                # if not color == "(100, 100, 100)":
                #     continue
                polygons, segmentations = create_sub_mask_annotation(sub_mask)
                category_id = category_colors[color]

                # "annotations" info

                # Check if we have classes that are a multipolygon
                if category_id in multipolygon_ids:
                    # Combine the polygons to calculate the bounding box and area
                    multi_poly = MultiPolygon(polygons)
                                    
                    annotation = create_annotation_format(multi_poly, segmentations, image_id, category_id, annotation_id, keypoint=key_points)

                    annotations.append(annotation)
                    annotation_id += 1
                else:
                    for i in range(len(polygons)):
                        # Cleaner to recalculate this variable
                        segmentation = [np.array(polygons[i].exterior.coords).ravel().tolist()]
                        
                        annotation = create_annotation_format(polygons[i], segmentation, image_id, category_id, annotation_id, keypoint=key_points)
                        
                        annotations.append(annotation)
                        annotation_id += 1
            image_id += 1
    return images, annotations, annotation_id

if __name__ == "__main__":
    # Get the standard COCO JSON format
    coco_format = get_coco_json_format()
    
    mask_path = '/Users/kaejong/workspace/antigravity/test/image/'
    # save_mask_seperately(mask_path)

        
    # Create category section
    coco_format["categories"] = create_category_annotation(category_ids)
    
    # Create images and annotations sections
    coco_format["images"], coco_format["annotations"], annotation_cnt = images_annotations_info(mask_path)

    with open(os.path.join(mask_path,"pad.json"),"w") as outfile:
        json.dump(coco_format, outfile)
        
    #     print("Created %d annotations for images in folder: %s" % (annotation_cnt, mask_path))

    #     with open("pad.json","r") as outfile:
    #         anno = json.load(outfile)
