import os



data_txt_dir = 'test.txt'
data_label_dir = 'TE.txt'
output_file = 'spilted'+'_'+data_txt_dir

if __name__ ==  '__main__':
    label_txt = open(os.path.join('/home1/food/VireoFood172/SplitAndIngreLabel/', data_label_dir))
    line =label_txt.readline()
    i=0
    label_dict={}
    output_line = []
    while line:
        if '/' in line:
            temp_line_pre = int(line[1:line.rfind('/', 1)])-1
            temp_line_next = line[1:-1]
            output_line.append(temp_line_next + ' ' + str(temp_line_pre))
            i+=1
            if i%100 == 0:
                print(i)
        line = label_txt.readline()

    with open(os.path.join('/home1/food/VireoFood172/SplitAndIngreLabel', output_file),'w') as f:
        for j in output_line:
            f.write(str(j)+'\n')
            pass


