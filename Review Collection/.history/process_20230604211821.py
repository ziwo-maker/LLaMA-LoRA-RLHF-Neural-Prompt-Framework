from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import logging
import json
logging.basicConfig(filename="phoenix.log",level = logging.INFO, filemode="a", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")

logger = logging.getLogger(__name__)

logger.info("success")
filename=["billa/billa_lcsts.json",
          'chatgpt/lcsts.json',
          'chatglm/lcsts_chatglm.json',
          'phoenix/phoenix_lcsts.json']
for k in filename:
    len_all=0;
    with open (k,'r',encoding="utf-8") as f:
        super_data=json.load(f); 
    for i in super_data:
        len_all+=len(i)
    len_all=len_all*1.0/(len(super_data))
    logger.info('len of '+k)
    logger.info(len_all)