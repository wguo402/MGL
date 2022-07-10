import os
from shutil import copyfile
import torch
from torch.utils import data
from PIL import Image


def My_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(data.Dataset):

    def __init__(self,imagepath, txt_dir,save_path):
        self.image_path = imagepath
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
        if '/' in img_name:
            real_name = img_name[img_name.rfind('/', 1):][1:]
            label_name = img_name[:img_name.rfind('/', 1)]
            src_path = self.image_path + label_name + '/' + real_name

        else:
            real_name = img_name
            src_path = image_path  + '/' + real_name



        # if label_name < 10:
        #     label_name = "00%d" % label_name
        # elif label_name < 100:
        #     label_name = "0%d" % label_name
        # else:
        #     label_name = str(label_name)

        try:
            img = self.loader(src_path)

            dst_path_temp = self.save_path + '/' + str(label)

            if not os.path.isdir(dst_path_temp):
                os.mkdir(dst_path_temp)
            copyfile(src_path, dst_path_temp + '/' + real_name)
            if index%100 ==0:
                print(index)
        except:
            print('erro picture:', img_name)



        return 1


image_path = '/home1/food/ChineseFoodNet/release_data/test/'
image_path_train = '/home1/food/ChineseFoodNet/release_data/train/'
DIR_TRAIN_IMAGES = '/home1/food/ChineseFoodNet/release_data/few_short_train.txt'
DIR_TEST_IMAGES = '/home1/food/ChineseFoodNet/release_data/few_short_test.txt'

train_save_path = '/home1/food/ChineseFoodNet/release_data/pytorch/train'
test_save_path = '/home1/food/ChineseFoodNet/release_data/pytorch/test'

generate_test_ataset = MyDataset(imagepath=image_path_train,txt_dir=DIR_TEST_IMAGES,save_path=test_save_path)
generate_train_dataset=MyDataset(imagepath=image_path_train,txt_dir=DIR_TRAIN_IMAGES,save_path=train_save_path)

if not os.path.isdir(train_save_path):
    os.mkdir(train_save_path)
# if not os.path.isdir(val_save_path):
#         os.mkdir(val_save_path)
if not os.path.isdir(test_save_path):
    os.mkdir(test_save_path)

for i,j in enumerate(generate_train_dataset):
    pass
for i,j in enumerate(generate_test_ataset):
    pass





# for i,j in enumerate(generate_val_dataset):
#     pass





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
