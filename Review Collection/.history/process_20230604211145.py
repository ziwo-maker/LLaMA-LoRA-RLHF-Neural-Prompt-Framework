import json
filename="billa/billa_lcsts.json"
len_all=0;
with open (filename,'r',encoding="utf-8") as f:
    super_data=json.load(f); 
    for i in super_data:
        len_all+=len(i)