import xlrd
import sys
import logging
import json
import xlwt;

filename=["billa/billa_lcsts.json",
          'billa/billa_lcsts2.json',
          'billa/billa_lcsts3.json',
          'chatglm/lcsts_chatglm.json',
          'chatglm/chatglm_lcsts2.json',
          'chatglm/chatglm_lcsts3.json',
          'phoenix/phoenix_lcsts.json',
          'phoenix/phoenix_lcsts2.json',
          'phoenix/phoenix_lcsts3.json',
          
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
        print('******\n')
            
    len_all=len_all*1.0/count

   
worksheet.write(0,0, "")