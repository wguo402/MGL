import os
import random

l_train=[]
l_val=[]
l_test=[]


def ReadFileDatas(original_filename):
    file=open(original_filename,'r+')
    FileNameList=file.readlines()
    random.shuffle(FileNameList)
    file.close()
    print("数据集总量：", len(FileNameList))
    return FileNameList

def TrainValTestFile(FileNameList):
    i=0
    j=len(FileNameList)
    for line in FileNameList:
        if i<(j*0.8):
            i+=1
            l_train.append(line)
        elif i<(j*0.8):
            i+=1
            l_val.append(line)
        else:
            i+=1
            l_test.append(line)
    return l_train,l_val,l_test

def WriteDatasToFile(listInfo, new_filename):
    file_handle = open(new_filename,'w')
    for str_Result in listInfo:
        file_handle.write(str_Result)
    file_handle.close()
if __name__ == "__main__":
      listFileInfo = ReadFileDatas('/home1/food/food-101/spilted_train.txt') # 读取文件
      l_train,l_val,l_test=TrainValTestFile(listFileInfo)
      WriteDatasToFile(l_train, '/home1/food/food-101/split_t_t_v/all_train.txt')
      WriteDatasToFile(l_val, '/home1/food/food-101/split_t_t_v/all_val.txt')
      # WriteDatasToFile(l_test, '/home1/food/food-101/split_t_t_v/all_test.txt')