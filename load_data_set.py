from torchvision import transforms as T
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder




def load_data(path,transforms):
    dataset = ImageFolder(path,transforms)

# cat文件夹的图片对应label 0，dog对应1
    #print(dataset.class_to_idx)
    return dataset

# 所有图片的路径和对应的label
#print(dataset.imgs)

# 没有任何的transform，所以返回的还是PIL Image对象
#print(dataset[0][1])# 第一维是第几张图，第二维为1返回label
#print(dataset[0][0]) # 为0返回图片数据
# plt.imshow(dataset[0][0])
# plt.axis('off')
# plt.show()
