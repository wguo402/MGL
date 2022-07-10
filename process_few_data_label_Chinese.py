import os
import random


data_txt_dir = 'train_list.txt'

output_file = 'few_short'+'_'+'train.txt'
output_file_test = 'few_short'+'_'+'test.txt'

data_test_txt_dir = 'test_truth_list.txt'

if __name__ ==  '__main__':

    i=0
    label_dict={}
    for i in range(208):
        label_dict[str(i)]=i

    data_txt = open(os.path.join('/home1/food/ChineseFoodNet/release_data', data_txt_dir))
    data_line = data_txt.readline()

    data_txt_test = open(os.path.join('/home1/food/ChineseFoodNet/release_data', data_test_txt_dir))
    data_line_test = data_txt_test.readline()

    train_output_line=[]
    test_output_line = []
    dict_key = random.sample(label_dict.keys(), 50)
    test_dict={}
    for i,j in enumerate(dict_key):
         test_dict[j]=i
         print(label_dict[j])
         del label_dict[j]
    dict_key = random.sample(label_dict.keys(), 158)
    train_dict={}
    for i,j in enumerate(dict_key):
        train_dict[j] = i
        print(label_dict[j])
        del label_dict[j]

    while data_line_test:
        temp_line = data_line_test[data_line_test.rfind(' ', 1):-1][1:]
        name = data_line_test[0:data_line_test.rfind(' ', 1)]
        # print(str(label_dict[temp_line]))
        if temp_line in train_dict:
            train_output_line.append(name+' '+str(train_dict[temp_line]))
        else:
            test_output_line.append(name + ' ' + str(test_dict[temp_line]))
        data_line_test = data_txt_test.readline()
        # print(output_line)

    while data_line:
        temp_line = str(int(data_line[0:data_line.rfind('/', 1)]))
        name = data_line[0:data_line.rfind(' ', 1)]
        # print(str(label_dict[temp_line]))
        if temp_line in train_dict:
            train_output_line.append(name+' '+str(train_dict[temp_line]))
        else:
            test_output_line.append(name + ' ' + str(test_dict[temp_line]))
        data_line = data_txt.readline()
        # print(output_line)

    with open(os.path.join('/home1/food/ChineseFoodNet/release_data', output_file),'w') as f:
        for j in train_output_line:
            f.write(str(j)+'\n')
            pass

    with open(os.path.join('/home1/food/ChineseFoodNet/release_data', output_file_test),'w') as f:
        for j in test_output_line:
            f.write(str(j)+'\n')
            pass
