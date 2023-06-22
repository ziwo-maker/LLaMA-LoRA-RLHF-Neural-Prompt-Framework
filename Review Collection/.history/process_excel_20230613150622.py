import xlrd
import sys
import logging
import json
import xlwt;

filename=["chatglm/generation_chatglm_saved.json",
          "moss/moss_save_generation.json",
          "mpt/mpt_save_generation.json",
          "billa/billa_save_generation.json",
          "phoenix/phoenix_save_generation.json",

          "chinensealpaca/generation_chinesealpaca_saved.json",
          ]


workbook = xlwt.Workbook(encoding= 'ascii')

    
worksheet = workbook.add_sheet("Sheet1")

for j,k in enumerate(filename):
    len_all=0;
    count=0;
    max_len=0
    seq=''
    with open (k,'r',encoding="utf-8") as f:
        data_line=f.readline();
        data_line=json.loads(data_line)
        for i in data_line:
            worksheet.write(j,count,len(i) )
            count+=1;
            if(len(i)>max_len):
                max_len=len(i)
                
                seq=i;
        print(seq)
        print(max_len)
        print('******\n')
            
    len_all=len_all*1.0/count
    
workbook.save("len_ofwall.xls")

   
