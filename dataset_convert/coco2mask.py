from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import os
from PIL import Image
from pathlib import Path
import sys
import argparse
from arguments import parse_args

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

def coco_visualization(args):
    annFile = os.path.join(args.root_path, 'pad.json')

    coco = COCO(annFile)
    catIds = [0, 1]
    ids = coco.getImgIds(catIds=catIds)

    for imgIds in ids:
        annIds = coco.getAnnIds(imgIds = imgIds, catIds=catIds)
        anns = coco.loadAnns(annIds)
        print(anns)
        imgInfo = coco.loadImgs(imgIds)
        print(imgInfo)
        image_name = os.path.join(args.root_path, 'images', imgInfo[0]['file_name'])
        # image_name = os.path.join(root_path, imgInfo[0]['file_name'])
        image = Image.open(image_name).convert('RGB')
        plt.imshow(image)
        coco.showAnns(anns)
        plt.show()


if __name__ == "__main__":
    args = parse_args()
    coco_visualization(args)


# I = io.imread(os.path.join('/home/ljj/workspace/antigravity/snuailab_dev/data/pad', img['file_name']))


# # plt.axis('off')
# # plt.imshow(I)
# # plt.show()

# plt.imshow(I)
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns, draw_bbox=True)
# plt.show()