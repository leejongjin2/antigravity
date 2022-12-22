import os
import glob
import shutil

root_path = '/home/ljj/dataset/datasets_2022_12_21'

# images = glob.glob(os.path.join(root_path, 'images/*'))
# segs = glob.glob(os.path.join(root_path, 'segmentation/*'))
# depths = glob.glob(os.path.join(root_path, 'depths/*'))
# npys = glob.glob(os.path.join(root_path, 'npy/*'))

image_path = (os.path.join(root_path, 'images'))
seg_path = (os.path.join(root_path, 'segmentations'))
depth_path = (os.path.join(root_path, 'depths'))
npy_path = (os.path.join(root_path, 'npy'))
images = glob.glob(image_path+'/*')
segs = glob.glob(seg_path+'/*')
depths = glob.glob(depth_path+'/*')
npys = glob.glob(npy_path+'/*')

length_lst = [len(images), len(segs), len(depths), len(npys)]
print(length_lst) 
format_lst = ['img', 'mask', 'depth']


def check_name(product_name, product_number,type_path, type, form='.png'):
    if not os.path.isfile(os.path.join(type_path, product_name+'_'+type+'_'+product_number+form)):
        for src_file in glob.glob(os.path.join(type_path, product_name+'*')):
            if product_number in os.path.basename(src_file):
                dst_file = os.path.join(type_path, product_name + '_'+type+'_' + product_number+form)
                shutil.move(src_file, dst_file)
                print(src_file, dst_file)

# print(images)
for img_name in images:
    img_name_ = os.path.basename(img_name)
    a = img_name_.split('_')
    product_name = a[0]+'_'+a[1]
    product_number = a[3].split('.')[0]
    if 'img' != format_lst[0]:
        print(product_name)
    check_name(product_name, product_number, seg_path, 'mask')
    check_name(product_name, product_number, depth_path, 'depth')
    check_name(product_name, product_number, npy_path, 'depth', '.npy')