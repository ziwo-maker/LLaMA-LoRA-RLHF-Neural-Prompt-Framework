from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()



from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import logging
model_path = "BelleGroup/BELLE-7B-2M" # You can modify the path for storing the local model
model =  AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
logging.basicConfig(filename='belleblog.txt',
                     format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s-%(funcName)s',
                     level=logging.ERROR)


logger=logging.getLogger();


save_path='./seconddata'
data_path='./seconddata'
modelname='chatglm'
def get_value(inputs):
        response, history = model.chat(tokenizer, inputs, history=[])
        return response;


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
        s=i['question']
        tmp=get_value(s,path)
        tmp=tmp+"正确答案是："+i["answer"]+'\n'
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
    path1=data_path+'/idom/chid/dataclean.json'
    path2=save_path+'/idom/chid/'+modelname+'_chid.json'
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
        s=s+i['content']
        
        tmp=get_value(s,path)
        tmp=tmp+"正确答案是"+i['groundTruth'][0]
        ans.append(tmp)
        
        print("正确答案是"+i['groundTruth'][0])
        if(i['groundTruth'][0] in tmp):
            right=right+1
            print("此题正确")
        else:
            wrong=wrong+1
            print("此题错误")
    print(right*1.0/(right+wrong))
    logger.info(right+wrong)
    with open(path2,'w') as f:
        json.dump(ans,f);
        
do_child();
do_generation();
do_c3();
def do_c3():
    path1=data_path+'/MRDL/c3/dataclean.json'
    path2=save_path+'/MRDL/c3/'+modelname+'_chid.json'
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
        
        tmp=get_value(s,path)
        ans.append(tmp)
        
        print("正确答案是"+i[1][0]["answer"])
        if(i[1][0]["answer"] in tmp):
            right=right+1
            print("此题正确")
        else:
            wrong=wrong+1
            print("此题错误")
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
        s=i["original_text"];
        tmp=get_value(s,path)
        
        print("正确答案是"+i["ans"])
        if(i["ans"] in tmp):
            right=right+1
            print("此题正确")
        else:
            wrong=wrong+1
            print("此题错误")
        ans.append(tmp)
    logger.info(right*1.0/(right+wrong))
    print(right+wrong)
    with open(path2,'w') as f:
        json.dump(ans,f);
        
do_math23();
def do_lcsts():
    path1=data_path+'/math/math23/dataclean.json'
    path2=save_path+'/math/math23/'+modelname+'_lcsts.json'
    with open(path1,'r') as f:
            data_all=json.load(f)

    ans=[]
    for i in data_all:
        s="请将这句话提取出一个标题"+i["content"];
        
        tmp=get_value(s,path)
        tmp=tmp+i['title']
        ans.append(tmp)
    with open(path2,'w') as f:
        json.dump(ans,f);

def do_lcsts2():
    path1=data_path+'/TextSummary/lcstc/dataclean.json'
    path2=save_path+'+modelname'+'_lcsts2.json'
    with open(path1,'r') as f:
            data_all=json.load(f)

    ans=[]
    for i in data_all:
        s="你现在是一个新闻工作者，请将这段话，提取出一个标题"+i["content"];
        
        tmp=get_value(s,path)
        tmp=tmp+i['title']
        ans.append(tmp)
    with open(path2,'w') as f:
        json.dump(ans,f);
    
    
#ape任务
#GPT 0.23正确率吧
def do_ape():
    path1=data_path+'/math/ape210k/dataclean.json'
    path2=save_path+'/math/ape210k/'+modelname+'_lcsts.json'
    with open(path1,'r') as f:
            data_all=json.load(f)
    ans=[]
    right=0;
    wrong=0;
    for i in data_all:
        s=i["segmented_text"];
        s=s+",只给出答案"
        tmp=get_value(s,path)
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
    print("acc")
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
#         tmp=get_value(s,path)
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
    path1=data_path+'/math/ape210k/dataclean.json'
    path2=save_path+'/math/ape210k/'+modelname+'_lcsts.json'
    with open('/kaggle/input/recodata/dataclean.json','r') as f:
            data_all=json.load(f)
  

    ans=[]
    right=0;
    wrong=0;
    data_all=data_all[0]
    for i in data_all:
        s="阅读这段话，"
        s+=i['query'];
        s=s+",只给出答案+\n"+i['alternatives']+'\n'+i['passage']
        tmp=get_value(s,path)
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
    print("acc")
    logger.info((right*1.0)/(wrong+right))
    with open(path2,'w') as f:
        json.dump(ans,f);  
        
do_ape()
do_reco()
do_lcsts2()
