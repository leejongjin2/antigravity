from pycocotools.coco import COCO
import numpy as np 
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from PIL import Image

# annFile = '/home/ljj/workspace/antigravity/snuailab_dev/data/pad_sample/pad.json'

# root_path = '/home/ljj/workspace/anti/HRNet-for-Fashion-Landmark-Estimation.PyTorch/data/deepfashion2/validation'

root_path = '/home/ljj/dataset/anti_sample/train'
# /home/ljj/workspace/antigravity/snuailab_dev/data/pad_sample'
# annFile = os.path.join(root_path, 'val-coco_style.json')
annFile = os.path.join(root_path, 'pad.json')

# imgFile = os.path.join(root_path, 'image', '000001.jpg')
coco = COCO(annFile)
catIds = coco.getCatIds(catNms=['Short sleeve top'])
ids = coco.getImgIds(catIds=catIds)

for imgIds in ids:
    annIds = coco.getAnnIds(imgIds = imgIds, catIds=catIds)
    anns = coco.loadAnns(annIds)
    print(anns)
    imgInfo = coco.loadImgs(imgIds)
    print(imgInfo)
    image_name = os.path.join(root_path, 'images', imgInfo[0]['file_name'])
    # image_name = os.path.join(root_path, imgInfo[0]['file_name'])
    image = Image.open(image_name).convert('RGB')
    plt.imshow(image)
    coco.showAnns(anns)
    plt.show()

# I = io.imread(os.path.join('/home/ljj/workspace/antigravity/snuailab_dev/data/pad', img['file_name']))


# # plt.axis('off')
# # plt.imshow(I)
# # plt.show()

# plt.imshow(I)
# annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
# anns = coco.loadAnns(annIds)
# coco.showAnns(anns, draw_bbox=True)
# plt.show()