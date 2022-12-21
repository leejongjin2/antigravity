import glob 
root_path='/home/ljj/workspace/antigravity/sample/80d'
data_lst = glob.glob(root_path+"*[rgb]*|*[mask]*")

print(data_lst)
