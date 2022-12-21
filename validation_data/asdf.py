import glob
import os
import shutil

root_path = '/home/ljj/dataset/chec'
image_path = (os.path.join(root_path, 'images'))
seg_path = (os.path.join(root_path, 'segmentations'))
depth_path = (os.path.join(root_path, 'depths'))
npy_path = (os.path.join(root_path, 'npy'))
images = glob.glob(image_path+'/*')
segs = glob.glob(seg_path+'/*')
depths = glob.glob(depth_path+'/*')
npys = glob.glob(npy_path+'/*')


def delete_underbar(img_name):
    img_name_ = os.path.basename(img_name)
    a = img_name_.split('_')
    if len(a)>4 and 'shoo' in a[0]:
        parent_path = (os.path.dirname(img_name))
        rename = a[0]+a[1]+'_'+a[2]+'_'+a[3]+'_'+a[4]#+'_'+a[5]
        dst_name = os.path.join(parent_path, rename)
        print(img_name)
        print(dst_name)
        shutil.move(img_name, dst_name) 

# for img_name in images:
#     delete_underbar(img_name)
# for mask_name in segs:
#     delete_underbar(mask_name)
# for depth_name in depths:
#     delete_underbar(depth_name)
for npy_name in npys:
    delete_underbar(npy_name)