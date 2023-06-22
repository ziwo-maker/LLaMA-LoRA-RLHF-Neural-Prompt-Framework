import json 


filename='白杨叶甲.txt';
with open(filename,'r',encoding='gbk') as f:
    data_line=f.readline();
    print(data_line)