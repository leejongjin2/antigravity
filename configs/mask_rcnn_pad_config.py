# The new config inherits a base config to highlight the necessary modification
_base_ = '../mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('pad',)
data = dict(
    train=dict(
        img_prefix='data/pad/',
        classes=classes,
        ann_file='data/pad/pad.json'),
    val=dict(
        img_prefix='data/pad_sample/',
        classes=classes,
        ann_file='data/pad_sample/pad.json'),
    # test=dict(
    #     img_prefix='data/balloon/val/',
    #     classes=classes,
    #     ann_file='data/balloon/val/annotation_coco.json')
    )

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'ckpt/mask_rcnn_test.pth'
