import json 


filename='白杨叶甲.txt';
with open(filename,'r',encoding='utf-8') as f:
    data_line=f.readlines();
    for i in data_line:
        j=i.split('|')
        print(j)
        print()