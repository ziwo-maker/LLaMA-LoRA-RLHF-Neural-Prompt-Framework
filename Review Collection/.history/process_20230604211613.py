from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import logging
logging.basicConfig(filename="phoenix.log",level = logging.INFO, filemode="a", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")

logger = logging.getLogger(__name__)

logger.info("success")
filename=["billa/billa_lcsts.json",
          'chatgpt/lcsts.json',
          'chatglm/lcsts_chatglm.json',
          'phoenix/phoenix_lcsts.json']
len_all=0;
with open (filename,'r',encoding="utf-8") as f:
    super_data=json.load(f); 
for i in super_data:
    len_all+=len(i)
print(len_all)