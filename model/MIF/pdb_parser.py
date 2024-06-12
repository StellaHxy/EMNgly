import os



filename = '5xco'
single_path = 'examples/keras/single/'+filename+'_single.pdb'
origin_path = 'examples/keras/'+filename+'.pdb'

def test():
    with open(origin_path, 'r') as pdbfile:
        lines = pdbfile.readlines()
    for line in lines:
        print(line)


def generate2():
    line_list =[]
    with open(single_path, 'w+') as f1:
        with open(origin_path , 'r') as pdbfile:
            lines = pdbfile.readlines()
            for i, line in enumerate(lines):
                if line[:4] == 'ATOM' or line[:6] == "HETATM":
                    # print (line)
                    # Split the line
                    splitted_line = [line[:6], line[6:11], line[11] ,line[12:16], line[16],line[17:20],line[20], line[21] ,line[22:26], line[26:30],line[30:38], line[38:46], line[46:]]
                    print(splitted_line)
                    line_list.append(splitted_line)
    # 找出A链最后一个aa的序号
    value_end = 0
    for i, line in enumerate(line_list):
        if line[7] ==  'A' and line_list[i+1][7] != 'A':
            value_end = eval(line[8])
    print(value_end)

    # 修改B链的序号，并将链的名字改为A
    value_start = value_end + 20

    for i, line in enumerate(line_list):
        if line[7] != 'A':
            line_list[i][8] = eval(line[8]) + value_start
            line[7] = 'A'
            if line_list[i][8] > 0 and line_list[i][8] < 10:
                str1 = '   ' + str(line_list[i][8])
            elif line_list[i][8]>=10 and line_list[i][8] < 100:
                str1 = '  ' + str(line_list[i][8])
            elif line_list[i][8] >= 100 and line_list[i][8] < 1000 :
                str1 = ' ' + str(line_list[i][8])
            else:
                str1 = str(line_list[i][8])
            line_list[i][8] = str1
        print(line_list[i])

    with open(single_path , 'a+') as f2:
        lines = [''.join(i) for i in line_list]
        f2.writelines(lines)

def run():
    with open(single_path, 'w+') as f1:
        with open(origin_path , 'r') as pdbfile:
            lines = pdbfile.readlines()
            A_end = 0
            num = 0
            for i, line in enumerate(lines):
                if line[:4] == 'ATOM' or line[:6] == "HETATM":
                    line1 = line
                    # print (line)
                    # Split the line
                    # splitted_line = [line[:6], line[6:11], line[12:16], line[17:20], line[21], line[22:26], line[30:38], line[38:46], line[46:54]]
                    # print (splitted_line)
                    str0 = line1[22:26]
                    if line1[22:26] != lines[i-1][22:26]:
                        num += 1

                    # for i in range(0, 4):
                    #     if str0[i] == ' ':
                    #         str0 = str0.replace(str0[i], str(0))
                    #
                    # num_origin = int(str0[3]) + int(str0[2]) * 10 + int(str0[1]) * 100 + int(str0[0]) * 1000

                    if not line1[21] == 'A':
                        # 判断该行是否为B链第一行，如果是，则获取A链末尾元素的值
                        if lines[i-1][21] == 'A':
                            str1 = lines[i-1][22:26]
                            print(str1)
                            for i in range(0, 4):
                                if str1[i] == ' ':
                                    str1 = str1.replace(str1[i], str(0))
                            print(str1)
                            A_end = int(str1[3]) + int(str1[2]) * 10 + int(str1[1]) * 100 + int(str1[0]) * 1000
                            print(A_end , type(A_end))
                            num = A_end + 20

                        # 判断该行是否为该原子是否与上一行的原子序号一致，如果不一致，说明位置改变，num+=1


                        if num > 0 and num < 100:
                            line1 = line1.replace(line1[24:26], str(num))
                        elif num >= 100 and num < 1000:
                            line1 = line1.replace(line1[23:26], str(num))
                        else:
                            line1 = line1.replace(line1[22:26], str(num))
                    print(line1)
                    f1.write(line1)


                    # To format again the pdb file with the fields extracted
                    # print ("%-6s%5s %4s %3s %s%4s    %8s%8s%8s\n"%tuple(splitted_line))

generate2()