import os
from shutil import copyfile
import torch
from torch.utils import data
from PIL import Image
# You only need to change this line to your dataset download path
# download_path = '/home1/jinyl/fusai/test'
#
# if not os.path.isdir(download_path):
#     print('please change the download_path')
#
# save_path = download_path + '/pytorch'
# if not os.path.isdir(save_path):
#     os.mkdir(save_path)
# #-----------------------------------------
#query
# query_path = download_path + '/query_a'
# query_save_path = download_path + '/pytorch/query'
# if not os.path.isdir(query_save_path):
#     os.mkdir(query_save_path)

# for root, dirs, files in os.walk(query_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='png':
#             continue
#         # ID  = name.split('_')
#         src_path = query_path + '/' + name
#         dst_path = query_save_path + '/' + name
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)

#-----------------------------------------
#multi-query
# query_path = download_path + '/gt_bbox'
# # for dukemtmc-reid, we do not need multi-query
# if os.path.isdir(query_path):
#     query_save_path = download_path + '/pytorch/multi-query'
#     if not os.path.isdir(query_save_path):
#         os.mkdir(query_save_path)
#
#     for root, dirs, files in os.walk(query_path, topdown=True):
#         for name in files:
#             if not name[-3:]=='jpg':
#                 continue
#             ID  = name.split('_')
#             src_path = query_path + '/' + name
#             dst_path = query_save_path + '/' + ID[0]
#             if not os.path.isdir(dst_path):
#                 os.mkdir(dst_path)
#             copyfile(src_path, dst_path + '/' + name)

#-----------------------------------------
#gallery
# gallery_path = download_path + '/gallery_a'
# gallery_save_path = download_path + '/pytorch/gallery'
# if not os.path.isdir(gallery_save_path):
#     os.mkdir(gallery_save_path)
#
# for root, dirs, files in os.walk(gallery_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='png':
#             continue
#         # ID  = name.split('_')
#         src_path = gallery_path + '/' + name
#         dst_path = gallery_save_path + '/' + name
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)


def My_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(data.Dataset):

    def __init__(self, txt_dir,save_path):
        self.image_path = image_path
        self.save_path = save_path
        self.loader = My_loader
        data_txt = open(txt_dir, 'r')
        imgs = []

        for line in data_txt:
            line = line.strip()
            words = line.split()

            imgs.append((words[0], int(words[1])))

        self.imgs = imgs



    def __len__(self):

        return len(self.imgs)

    def __getitem__(self, index):
        img_name, label = self.imgs[index]
        src_path = self.image_path + img_name

        # img = self.loader(src_path)
        try:
            img = self.loader(src_path)
            temp_line = img_name[img_name.rfind('/', 1)+1:]
            dst_path_temp = self.save_path + '/' + str(label)
            dst_path = self.save_path + '/'
            if not os.path.isdir(dst_path_temp):
                os.mkdir(dst_path_temp)
            copyfile(src_path,dst_path_temp + '/' + temp_line)
            if index%100 ==0:
                print(index)
        except:
            print('erro picture:', img_name)



        return 1


# DIR_TRAIN_IMAGES = '/home1/food/isiaFood_200/train_finetune_v2.txt'
# DIR_val_IMAGES = '/home1/food/isiaFood_200/val_finetune_v2.txt'
# DIR_TEST_IMAGES = '/home1/food/isiaFood_200/test_finetune_v2.txt'
# train_save_path = '/home1/food/isiaFood_200/pytorch/train'
# val_save_path = '/home1/food/isiaFood_200/pytorch/val'
# test_save_path = '/home1/food/isiaFood_200/pytorch/test'
# image_path = '/home1/food/food-101/images/'
# DIR_TRAIN_IMAGES = '/home1/food/food-101/few_short_train.txt'
# DIR_TEST_IMAGES = '/home1/food/food-101/few_short_test.txt'
# train_save_path = '/home1/food/food-101/pytorch/train'
# test_save_path = '/home1/food/food-101/pytorch/test'

image_path = '/home1/food/VireoFood172/ready_chinese_food/'
DIR_TRAIN_IMAGES = '/home1/food/VireoFood172/SplitAndIngreLabel/few_short_train.txt'
DIR_TEST_IMAGES = '/home1/food/VireoFood172/SplitAndIngreLabel/few_short_test.txt'
train_save_path = '/home1/food/VireoFood172/pytorch/train'
test_save_path = '/home1/food/VireoFood172/pytorch/test'



generate_train_dataset = MyDataset(txt_dir=DIR_TRAIN_IMAGES,save_path=train_save_path)
# generate_val_dataset = MyDataset(txt_dir=DIR_val_IMAGES,save_path=val_save_path)
generate_test_ataset = MyDataset(txt_dir=DIR_TEST_IMAGES,save_path=test_save_path)

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
# if not os.path.isdir(val_save_path):
#         os.mkdir(val_save_path)
if not os.path.isdir(test_save_path):
    os.mkdir(test_save_path)
for i,j in enumerate(generate_train_dataset):
    pass
# for i,j in enumerate(generate_val_dataset):
#     pass
for i,j in enumerate(generate_test_ataset):
    pass




#---------------------------------------
#train_all
# train_path = download_path + '/bounding_box_train'
# train_save_path = download_path + '/pytorch/train_all'
# if not os.path.isdir(train_save_path):
#     os.mkdir(train_save_path)
#
# for root, dirs, files in os.walk(train_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='jpg':
#             continue
#         ID  = name.split('_')
#         src_path = train_path + '/' + name
#         dst_path = train_save_path + '/' + ID[0]
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)


#---------------------------------------
#train_val
# train_path = download_path + '/bounding_box_train'
# train_save_path = download_path + '/pytorch/train'
# val_save_path = download_path + '/pytorch/val'
# if not os.path.isdir(train_save_path):
#     os.mkdir(train_save_path)
#     os.mkdir(val_save_path)
#
# for root, dirs, files in os.walk(train_path, topdown=True):
#     for name in files:
#         if not name[-3:]=='jpg':
#             continue
#         ID  = name.split('_')
#         src_path = train_path + '/' + name
#         dst_path = train_save_path + '/' + ID[0]
#         if not os.path.isdir(dst_path):
#             os.mkdir(dst_path)
#             dst_path = val_save_path + '/' + ID[0]  #first image is used as val image
#             os.mkdir(dst_path)
#         copyfile(src_path, dst_path + '/' + name)
