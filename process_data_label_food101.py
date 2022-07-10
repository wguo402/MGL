import os
import random


data_txt_dir = 'train.txt'
data_label_dir = 'labels.txt'
output_file = 'few_short'+'_'+data_txt_dir
output_file_test = 'few_short'+'_'+'test.txt'

data_test_txt_dir = 'test.txt'

if __name__ ==  '__main__':
    label_txt = open(os.path.join('/home1/food/food-101', data_label_dir))
    line =label_txt.readline()
    i=0
    label_dict={}
    while line:
        if ' ' in line:
            temp_line_pre = line[0:line.rfind(' ', 1)]
            temp_line_next = line[line.rfind(' ', 1) + 1:]
            if ' ' in temp_line_pre:
                sub_temp_line_pre = temp_line_pre[0:temp_line_pre.rfind(' ', 1)]
                sub_temp_line_next = temp_line_pre[temp_line_pre.rfind(' ', 1)+1:]
                if ' ' in sub_temp_line_pre:
                    sub_sub_temp_line_pre = sub_temp_line_pre[0:sub_temp_line_pre.rfind(' ', 1)]
                    sub_sub_temp_line_next = sub_temp_line_pre[sub_temp_line_pre.rfind(' ', 1) + 1:]
                    sub_temp_line_pre = sub_sub_temp_line_pre.lower()+'_'+sub_sub_temp_line_next.lower()
                temp_line_pre = sub_temp_line_pre.lower() + '_' + sub_temp_line_next.lower()
            temp_all_line = temp_line_pre.lower()+'_'+temp_line_next[:-1]
        else:
            temp_all_line = line[:-1].lower()
        label_dict[temp_all_line]=i
        i=i+1
        line = label_txt.readline()

    data_txt = open(os.path.join('/home1/food/food-101', data_txt_dir))
    data_line = data_txt.readline()

    data_txt_test = open(os.path.join('/home1/food/food-101', data_test_txt_dir))
    data_line_test = data_txt_test.readline()

    train_output_line=[]
    test_output_line = []
    dict_key = random.sample(label_dict.keys(), 30)
    test_dict={}
    for i,j in enumerate(dict_key):
         test_dict[j]=i
         print(label_dict[j])
         del label_dict[j]
    dict_key = random.sample(label_dict.keys(), 71)
    train_dict={}
    for i,j in enumerate(dict_key):
        train_dict[j] = i
        print(label_dict[j])
        del label_dict[j]

    while data_line_test:
        temp_line = data_line_test[0:data_line_test.rfind('/', 1)]
        # print(str(label_dict[temp_line]))
        if temp_line in train_dict:
            train_output_line.append(data_line_test[:-1]+'.jpg'+' '+str(train_dict[temp_line]))
        else:
            test_output_line.append(data_line_test[:-1] + '.jpg' + ' ' + str(test_dict[temp_line]))
        data_line_test = data_txt_test.readline()
        # print(output_line)

    while data_line:
        temp_line = data_line[0:data_line.rfind('/', 1)]
        # print(str(label_dict[temp_line]))
        if temp_line in train_dict:
            train_output_line.append(data_line[:-1]+'.jpg'+' '+str(train_dict[temp_line]))
        else:
            test_output_line.append(data_line[:-1] + '.jpg' + ' ' + str(test_dict[temp_line]))
        data_line = data_txt.readline()
        # print(output_line)

    with open(os.path.join('/home1/food/food-101', output_file),'w') as f:
        for j in train_output_line:
            f.write(str(j)+'\n')
            pass

    with open(os.path.join('/home1/food/food-101', output_file_test),'w') as f:
        for j in test_output_line:
            f.write(str(j)+'\n')
            pass
