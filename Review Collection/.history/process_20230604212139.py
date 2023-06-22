from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import logging
import json
logging.basicConfig(filename="len.log",level = logging.INFO, filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")

logger = logging.getLogger(__name__)

logger.info("success")
filename=["billa/billa_lcsts.json",
          'chatgpt/lcsts.json',
          'chatglm/lcsts_chatglm.json',
          'phoenix/phoenix_lcsts.json']

for k in filename:
    len_all=0;
    count=0;
    with open (k,'r',encoding="utf-8") as f:
        data_line=f.readline();
        data_line=json.loads(data_line)
        count+=1
        len_all+=len(data_line)
    len_all=len_all*1.0/count
    logger.info('len of '+k)
    logger.info(len_all)