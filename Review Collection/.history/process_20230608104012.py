from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import logging
import json
logging.basicConfig(filename="len.log",level = logging.INFO, filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")

logger = logging.getLogger(__name__)

logger.info("success")
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


for k in filename:
    len_all=0;
    count=0;
    max_len=0
    seq=''
    with open (k,'r',encoding="utf-8") as f:
        data_line=f.readline();
        data_line=json.loads(data_line)
        for i in data_line:
            len_all+=len(i)
            count+=1;
            if(len(i)>max_len):
                max_len=len(i)
                
                seq=i;
        print(seq)
        print('******\n')
            
    len_all=len_all*1.0/count
    logger.info('len of '+k)
    logger.info(len_all)