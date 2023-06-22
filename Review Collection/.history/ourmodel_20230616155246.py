import os
import sys

import fire
import gradio as gr
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import sys
import logging
import json



logging.basicConfig(filename="ourmodel.log",level = logging.INFO, filemode="a", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S")

logger = logging.getLogger(__name__)
save_path='./seconddata'
data_path='./seconddata'
modelname='ourmodel'
logger.info("success")

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    prompt_template: str = "",  # The prompt template to use, will default to alpaca.
    server_name: str = "0.0.0.0",  # Allows to listen on all interfaces by providing '0.
    share_gradio: bool = False,
    device: str ="cuda",
):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='huggyllama/llama-7b'"

    prompter = Prompter(prompt_template)
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def get_value(
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=4,
        max_new_tokens=128,
        stream_output=False,
        **kwargs,
    ):
        prompt = prompter.generate_prompt(instruction, input)
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )

        generate_params = {
            "input_ids": input_ids,
            "generation_config": generation_config,
            "return_dict_in_generate": True,
            "output_scores": True,
            "max_new_tokens": max_new_tokens,
        }

        if stream_output:
            # Stream the reply 1 token at a time.
            # This is based on the trick of using 'stopping_criteria' to create an iterator,
            # from https://github.com/oobabooga/text-generation-webui/blob/ad37f396fc8bcbab90e11ecf17c56c97bfbd4a9c/modules/text_generation.py#L216-L243.

            def generate_with_callback(callback=None, **kwargs):
                kwargs.setdefault(
                    "stopping_criteria", transformers.StoppingCriteriaList()
                )
                kwargs["stopping_criteria"].append(
                    Stream(callback_func=callback)
                )
                with torch.no_grad():
                    model.generate(**kwargs)

            def generate_with_streaming(**kwargs):
                return Iteratorize(
                    generate_with_callback, kwargs, callback=None
                )

            with generate_with_streaming(**generate_params) as generator:
                for output in generator:
                    # new_tokens = len(output) - len(input_ids[0])
                    decoded_output = tokenizer.decode(output)

                    if output[-1] in [tokenizer.eos_token_id]:
                        break

                    yield prompter.get_response(decoded_output)
            return  # early return for stream_output

        # Without streaming
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
        print(output)
        return output

    


        # testing code for readme
    def do_generation():
        path1=data_path+'/QA/WebQA/dataclean.json'
        path2=save_path+'/QA/WebQA/'+modelname+'_save_generation.json'
        with open(path1,'r') as f:
            data_all=json.load(f)
        ans=[];
        path2=save_path+'/QA/WebQA/'+modelname+'_save_generation.json'
        right=0;
        wrong=0
        for i in data_all:
            instruction=i['question']
            tmp=get_value(instruction)
            # tmp=tmp+"正确答案是："+i["answer"]+'\n'
            ans.append(tmp)
    #         print("正确答案是"+i['groundTruth'][0])
    #         if(i['groundTruth'][0] in tmp):
    #             right=right+1
    #             print("此题正确")
    #         else:
    #             wrong=wrong+1
    #             print("此题错误")
    #     print(right*1.0/(right+wrong))
    #     print(right+wrong)
        with open(path2,'w') as f:
            json.dump(ans,f);




    def do_child():
        path1=data_path+'/Idom/chid/dataclean.json'
        path2=save_path+'/Idom/chid/'+modelname+'_chid.json'
        with open(path1,'r') as f:
            data_all=json.load(f)
        ans=[];
        path='./chid_gpt_saved.json'
        right=0;
        wrong=0
        for i in data_all:
            s='阅读这段文字，'
            choices=''
            for t in range(len(i['candidates'])):
                for tj in range(len(i['candidates'][t])):

                    choices+=i['candidates'][t][tj]+','
            s="从选项:"+choices+"选择一个适当的词语，替换句中的#idiom#,回答只需包含你选择的选项\n"
            instruction=s+i['content']

            tmp=get_value(instruction)
            tmp=tmp+"正确答案是"+i['groundTruth'][0]
            ans.append(tmp)

            print("正确答案是"+i['groundTruth'][0])
            if(i['groundTruth'][0] in tmp):
                right=right+1
                print("此题正确")
            else:
                wrong=wrong+1
                print("此题错误")
        logger.info("acc of chid")
        print(right*1.0/(right+wrong))
        logger.info(right+wrong)
        with open(path2,'w') as f:
            json.dump(ans,f);

    # do_child();

    def do_c3():
        path1=data_path+'/MRDL/c3/dataclean.json'
        path2=save_path+'/MRDL/c3/'+modelname+'_c3.json'
        with open(path1,'r') as f:
            data_all=json.load(f)
        ans=[];

        right=0;
        wrong=0
        for i in data_all:
            s='阅读这段对话，'
            choices=''
            for t in range(len(i[1][0]['choice'])):
                choices+=i[1][0]['choice'][t]+','
            s=s+i[1][0]['question']+"从选项"+choices+"选择一个\n"


            for t in range(len(i[0])):
                s=s+i[0][t]
            instruction=s
            tmp=get_value(instruction)
            ans.append(tmp)

            print("正确答案是"+i[1][0]["answer"])
            if(i[1][0]["answer"] in tmp):
                right=right+1
                print("此题正确")
            else:
                wrong=wrong+1
                print("此题错误")
        logger.info("acc of c3")
        logger.info(right*1.0/(right+wrong))
        print(right+wrong)
        with open(path2,'w') as f:
            json.dump(ans,f);



    def do_math23():
        path1=data_path+'/math/math23/dataclean.json'
        path2=save_path+'/math/math23/'+modelname+'_math23.json'
        with open(path1,'r') as f:
            data_all=json.load(f)
        ans=[];

        right=0;
        wrong=0
        for i in data_all:
            instruction=i["original_text"];
            tmp=get_value(instruction)

            print("正确答案是"+i["ans"])
            if(i["ans"] in tmp):
                right=right+1
                print("此题正确")
            else:
                wrong=wrong+1
                print("此题错误")
            ans.append(tmp)
        logger.info("acc of math23")
        logger.info(right*1.0/(right+wrong))
        print(right+wrong)
        with open(path2,'w') as f:
            json.dump(ans,f);


    def do_lcsts():
        path1=data_path+'/TextSummary/lcstc/dataclean.json'
        path2=save_path+'/TextSummary/lcstc/'+modelname+'_lcsts.json'
        with open(path1,'r') as f:
                data_all=json.load(f)

        ans=[]
        for i in data_all:
            instruction="请将这句话提取出一个标题"+i["content"];

            tmp=get_value(instruction)
            tmp=tmp+i['title']
            ans.append(tmp)
        with open(path2,'w') as f:
            json.dump(ans,f);


    #ape任务
    #GPT 0.23正确率吧
    def do_ape():
        path1=data_path+'/math/ape210k/dataclean.json'
        path2=save_path+'/math/ape210k/'+modelname+'_ape.json'
        with open(path1,'r') as f:
                data_all=json.load(f)
        ans=[]
        right=0;
        wrong=0;
        for i in data_all:
            s=i["segmented_text"];
            instruction=s+",只给出答案"
            tmp=get_value(instruction)
            print(tmp)
            tmp=tmp+i['ans']
            print("正确答案是"+i['ans'])
            print(i['ans'] in tmp)
            if(i['ans'] in tmp):
                right=right+1;
                print("正确") 
            else:
                wrong=wrong+1;
                print("错误") 
            ans.append(tmp)
            print('\n')
        logger.info("acc of ape")
        logger.info((right*1.0)/(wrong+right))
        with open(path2,'w') as f:
            json.dump(ans,f);


    #doREADER任务
    # def do_read():
    #     with open('/kaggle/input/senconddata/seconddata/Reading_comprehension/dureader_checklist-data/dataclean.json','r') as f:
    #             data_all=json.load(f)
    #             data_all=data_all[0][0]['paragraphs'];
    #     path='/kaggle/working/dureader_gpt_saved.json'
    #     ans=[]
    #     right=0;
    #     wrong=0;
    #     for i in data_all:
    #         s=i['qas'][0]['question'];
    #         print(s)
    #         tmp=get_value(s)
    #         print(tmp)

    #         tmp=tmp+"正确答案是"+i['context']
    #         ans.append(tmp)
    #         print('\n')
    #     # print("acc")
    #     # print((right*1.0)/(wrong+right))
    #     with open(path,'w') as f:
    #         json.dump(ans,f);  




    #do_readerReCO任务
    #GPT 
    def do_reco():
        path1=data_path+'/Reading_comprehension/dataclean.json'
        path2=save_path+'/Reading_comprehension/'+modelname+'_reco.json'
        with open(path1,'r') as f:
                data_all=json.load(f)


        ans=[]
        right=0;
        wrong=0;
        data_all=data_all[0]
        for i in data_all:
            s="阅读这段话，"
            s+=i['query'];
            instruction=s+",只给出答案+\n"+i['alternatives']+'\n'+
            tmp=get_value(instruction,input=i['passage'])
            print(tmp)
            print("正确答案是"+i['answer'])
            tmp=tmp+i['answer']
            if(i['answer'] in tmp):
                right=right+1;
                print("正确") 
            else:
                wrong=wrong+1;
                print("错误") 
            ans.append(tmp)
            print("right+wrong:")
            print(right+wrong)
            if(right+wrong==100):
                break;
            print('\n')
        print("acc of rece ")
        logger.info("acc of reco")
        logger.info((right*1.0)/(wrong+right))
        with open(path2,'w') as f:
            json.dump(ans,f);  
    count=100;
    def do_CINLID():
        path1=data_path+'/Idom/CINLID/dataclean.json'
        path2=save_path+'/Idom/CINLID/'+modelname+'_CINLID.json'

        ans=[]
        right=0;
        wrong=0;
        filename='./seconddata/Idom/CINLID/Idiom_NLI.txt'
        with open (filename,'r',encoding="utf-8") as f:
            for i in range(0,count):
                data_line=f.readline();
                data_line=data_line.split('\t');
                s="下面两个成语是什么关系，相似，相反，还是不相关？只给出你的选择"
                instruction=s+data_line[0]+','+data_line[1];
                tmp=get_value(instruction)
                print(tmp)
                if(data_line[2]=='entailment\n' and '似' in tmp):
                    right+=1;
                    print("正确")
                elif(data_line[2]=='neutral\n' and '不' in tmp):
                    right+=1;
                    print("正确")
                elif(data_line[2]=='contradiction\n' and '反' in tmp):
                    right+=1;
                    print("正确")
                else:
                    wrong+=1;
                    print("错误")
                ans.append(tmp)
        print('\n')
        print("acc")
        print((right*1.0)/(wrong+right))
        with open(path2,'w') as f:
            json.dump(ans,f);
    # logger.info("begin generation")
    # do_generation();
    # logger.info("begin c3")
    # do_c3(); 
    # logger.info("begin math23")
    # do_math23();
    # logger.info("begin ape")
    # do_ape()
    # logger.info("begin reco")
    # do_reco()
    # logger.info("begin lcsts")
    # do_lcsts()
    # logger.info("begin chid")
    # do_child()
    logger.info("begin CINLID")
    do_CINLID()
    

if __name__ == "__main__":
    main()