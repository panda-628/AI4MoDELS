import json
import os
import csv
import time
import re
import pandas as pd
import shutil
import openai
import numpy as np
import datetime
import sys
import pandas as pd
import numpy as np
from collections import Counter
from Levenshtein import distance
from Levenshtein import ratio

from structure import *
from Parser import *
from config import *


def generate_relation_prompt(name,classes,description):
    prompt_list ={
        'prompt':'',
        'name':'',
    }
    message = []
    prompt = PROMPT_MODEL_RELATION.format(description,classes)
    message = [
        {"role": "user", "content": f"{prompt}"}    
    ]
    prompt_list['prompt'] = message
    prompt_list['name'] = name

    return prompt_list

def generate_inherit_relation_prompt(name,classes,description):
    prompt_list ={
        'prompt':'',
        'name':'',
    }
    message = []
    prompt = PROMPT_MODEL_INHERIT_RELATION.format(description,classes)
    message = [
        {"role": "user", "content": f"{prompt}"}    
    ]
    prompt_list['prompt'] = message
    prompt_list['name'] = name

    return prompt_list

# generate class 2-turn-conversation
def generate_pre_prompt(name,description):
    prompt_list ={
        'prompt':'',
        'name':'',
        }
    message = []
    prompt1 = PROMPT_MODEL_2_ROUND["prompt1"].format(description)
    prompt2 = PROMPT_MODEL_2_ROUND["prompt2"]
    message = [
        {"role": "system", "content":"As a professional software architect, you are creating a class model."},
        {"role":"user","content":f"{prompt1}"},
        {"role":"user","content":f"{prompt2}"}
    ]
    prompt_list['prompt'] = message
    prompt_list['name'] = name

    return prompt_list


# run llm to generate and get answer from LLM
def run_llm(prompt_list,llm,temperature,max_tokens,top_p,frequency_penalty,presence_penalty):
    log = []
    message = []
    prompt = prompt_list['prompt']

    os.environ['http_proxy'] = 'http://127.0.0.1:1080'
    os.environ['https_proxy'] = 'http://127.0.0.1:1080'
    
    openai.api_key=''
    # client = OpenAI(
        # base_url="https://api.gptsapi.net/v1",
        # api_key="sk-25Xdd89e6e1e84e7ea52378d7b82362253a53995eb52vUzn"
        # )
    message.append(prompt[0])
    for i in range(1,len(prompt)):
        message.append(prompt[i])
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = message,
            temperature = temperature,
            max_tokens = max_tokens,
            top_p = top_p,
            frequency_penalty = frequency_penalty,
            presence_penalty = presence_penalty,

            )
        User_message = prompt[i]['content']
        AI_answer = response.choices[0].message.content
        log.append(f'User:{User_message}\nAI:{AI_answer}')
        message.append({"role":"assistant","content":f'{AI_answer}'})
        log.append(f'\n')
        
    return AI_answer,log


def run_llm_relation(f,prompt_list,llm,temperature,max_tokens,top_p,frequency_penalty,presence_penalty):
    message = []
    prompt = prompt_list['prompt']
    
    openai.api_key=''
    # client = OpenAI(
        # base_url="https://api.gptsapi.net/v1",
        # api_key=""
        # )
    for i in range(len(prompt)):
        message.append(prompt[i])
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = message,
            temperature = temperature,
            max_tokens = max_tokens,
            top_p = top_p,
            frequency_penalty = frequency_penalty,
            presence_penalty = presence_penalty,

            )
        User_message = prompt[i]['content']
        AI_answer = response.choices[0].message.content
        #print("AI_answer:",AI_answer)
        print(f'{AI_answer}',file=f)
        message.append({"role":"assistant","content":f'{AI_answer}'})
        print(f'\n',file=f)
        
    return AI_answer




class StateMachineCSV:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.pre_result_list =[]
        self.base_result_list =[]
        self.pre_result=[]
        self.base_result=[]
        self.state = -1
        self.AI = 0
        self.cycle =1

    def process_csv(self):
        with open(self.csv_file, newline='') as csvfile:
            csv_reader = csv.reader(csvfile)
            # read log file by row
            for row in csv_reader:
                if row:
                    self._state_transition(','.join(i for i in row))
        return self.pre_result_list,self.base_result_list
    

    def _state_transition(self, row):
        if self.state ==-1:
            if f'---------------------{self.cycle}/30------' in row:
                self.state=0
                print(f'cycle--{self.cycle}')
                self.cycle+=1
                self.pre_result =[]
                self.base_result=[]
        elif self.state == 0:
            if "---------------------Prediction AI:" in row: 
                # 如果读取的这一行包含"Prediction AI:"，则进入状态1
                print(f'读取Prediction{row}')
                self.state = 1
            elif "----------------------Baseline AI:" in row:  
                # 如果读取的这一行包含"Baseline AI:"，则进入状态3
                print(f'读取Baseline{row}')
                self.state = 4
        elif self.state == 1:
            # 状态是一,
            # 读取到第1个"AI："时，加标志；
            if "AI:" in row:
                # 修改AI计数
                self.state = 2
        elif self.state ==2:
            # 读取到第2个"AI："时，加标志,保存行数据
            if "AI:" in row :
                # 删除行数据前"AI:"字符串
                row = row.partition("AI:")[2]
                # 保存行数据
                self.pre_result.append(row)
                self.state = 3
        elif self.state ==3:
            if "------------------------------------------------------------" in row:
                self.state = 6
            else:
                self.pre_result.append(row)
        elif self.state == 4:
            if "AI:" in row:
                row = row.partition("AI:")[2]
                self.base_result.append(row)
                self.state = 5
        elif self.state == 5:
                # 保存行数据
                if "------------------------------------------------------------" in row:
                    self.state = 7
                else:
                    self.base_result.append(row)
        elif self.state ==6:
            self.pre_result_list.append(self.pre_result)
            self.state=0
        elif self.state == 7:
            self.base_result_list.append(self.base_result)
            self.state=-1
        


def calculate(number_class_exact_match,number_class_generated,number_class_solution,number_attribute_exact_match,number_attribute_generated,number_attribute_solution):
    # class_precision
    
    if number_class_generated:
      class_precision = number_class_exact_match / number_class_generated
    else:
       class_precision = 0
    # class_recall
    if number_class_solution:
      class_recall = number_class_exact_match / number_class_solution
    else:
      class_recall = 0
    # class_f1
    if (class_precision + class_recall)==0:
       class_f1 = 0
    else:
      class_f1 = (2 * class_precision * class_recall) / (class_precision + class_recall)
      
    # attribute_precision
    if number_attribute_generated == 0 :
       attribute_precision = 0
    else:
      attribute_precision = number_attribute_exact_match / number_attribute_generated
    # attribute_recall
    if number_attribute_solution:
      attribute_recall = number_attribute_exact_match / number_attribute_solution
    else:
       attribute_recall = 0
    # attribute_f1
    if (attribute_precision + attribute_recall)==0:
       attribute_f1 = 0
    else:
      attribute_f1 = (2 * attribute_precision * attribute_recall) / (attribute_precision + attribute_recall)

    result = [class_precision,class_recall,class_f1,attribute_precision,attribute_recall,attribute_f1,number_class_exact_match,number_class_generated,number_class_solution,number_attribute_exact_match,number_attribute_generated,number_attribute_solution]
    return result



if __name__ == '__main__' :

    cycle = running_params['cycle']
    model_file = file['model_file']
    oracle_dataset = pd.read_csv(file['model_file'],encoding='latin1')

    name_list = oracle_dataset['Name']
    description_list = oracle_dataset['Description']
    oracle_classes_list = oracle_dataset['Classes']
    oracle_relationships_list = oracle_dataset['Associations']
    
    # ----------------------------------------
    # gpt3.5-tem0.6-2round-30cycle : 
    # experiment 2 every system log file
    # ----------------------------------------
    path ="E:/code/LLM_for_uml_new/experiment2-final/gpt3.5-tem0.6-2round-30cycle"


    all_score_file = f'{path}/Matched_all_score.csv' # a :final score of every system（baseline + our approach prediction）
    a = open(all_score_file,"w")
    cycle = running_params['cycle']
    print(f'system,{cycle} times,Ave_Test_Class_precision,\tAve_class_recall,\tAve_class_f1,\tAve_attribute_precision,\tAve_attribute_recall,\tAve_attribute_f1,',file=a)

    prediction_score_file = f'{path}/Matched_test_each_ex_score.csv' # ps : ours approach each system each cycle result
    ps=open(prediction_score_file,"w")
    print(f'system_name,cycle,Class_precision,class_recall,class_f1,attribute_precision,attribute_recall,attribute_f1,Class match, Class generate, Class oracle,Attribute match,Attribute generate, Attribute oracle',file=ps)

    baseline_score_file = f'{path}/Matched_baseline_each_ex_score.csv' # bs: baseline each system each cycle result
    bs=open(baseline_score_file,"w")
    print(f'system_name,cycle,Class_precision,class_recall,class_f1,attribute_precision,attribute_recall,attribute_f1,Class match, Class generate, Class oracle,Attribute match,Attribute generate, Attribute oracle',file=bs)
    os.chdir(path)
    new_folder_cls = 'MatchResult'
    os.makedirs(new_folder_cls)
    for name,oracle_classes in zip(name_list,oracle_classes_list):
        print(name)
        input_log_file = f'{path}/{name}.csv'
        print(input_log_file)
        sm = StateMachineCSV(input_log_file)
        pre_result_list,base_result_list = sm.process_csv()
        sum_test_result = [0,0,0,0,0,0]#代表本论文的方法
        sum_baseline_result = [0,0,0,0,0,0]
        ora_cls_parser = FileParser()
        oracle_classes,oracle_relationships = ora_cls_parser.parseLines(oracle_classes)
        cycle =30
        output_classes_file = f'{path}/1/{name}.csv'
        with open(output_classes_file,"w") as csvfile:
            writer = csv.writer(csvfile)
            log=[]
            for c,pre_AI_answer,base_AI_answer in zip(range(1,cycle+1),pre_result_list,base_result_list):
                pre_AI_answer = '\n'.join(row for row in pre_AI_answer)
                base_AI_answer = '\n'.join(row for row in base_AI_answer)
                log.append('-'*60)
                log.append('-'*60)
                log.append(f'---------------------{c}/{cycle}------{name}:')

                #prediction测试
                log.append('-'*60)
                log.append(f'---------------------Prediction AI:')
                
                class_info_parse = FileParser()
                generated_classes,relationships = class_info_parse.parseLines(pre_AI_answer)
                Pre_Ma = Matcher()
                matched_name = {}
                matched_class = {}
                unmatched_class = []
                
                #进行类和属性的的匹配
                matched_name,matched_class,unmatched_class,log_info = Pre_Ma.matchClasses(generated_classes,oracle_classes)
                log += log_info
                log.append('-'*60)
                pre_score = calculate(Pre_Ma.matched_classes_count,Pre_Ma.generated_classes_count,Pre_Ma.oracle_classes_count,Pre_Ma.matched_attributes_count,Pre_Ma.generated_attributes_count,Pre_Ma.oracle_attributes_count)
                print(f'-{c}/{cycle}  {name} Prediction Matching have done!')
                print(f'{name},{cycle},{pre_score[0]:4f},{pre_score[1]:4f},{pre_score[2]:4f},{pre_score[3]:4f},{pre_score[4]:4f},{pre_score[5]:4f},{Pre_Ma.matched_classes_count},{Pre_Ma.generated_classes_count},{Pre_Ma.oracle_classes_count},{Pre_Ma.matched_attributes_count},{Pre_Ma.generated_attributes_count},{Pre_Ma.oracle_attributes_count}',file=ps)

                
                baseline_classes,relationships = class_info_parse.parseLines(base_AI_answer)
                Base_Ma = Matcher()
                matched_name = {}
                matched_class = {}
                unmatched_class = []
                log.append('-'*60)
                log.append('-'*60)

                #prediction测试
                log.append('-'*60)
                log.append(f'---------------------Baseline AI:')
                #进行类和属性的的匹配

                matched_name,matched_class,unmatched_class,log_info = Base_Ma.matchClasses(baseline_classes,oracle_classes)
                log += log_info
                log.append('-'*60)
                base_score = calculate(Base_Ma.matched_classes_count,Base_Ma.generated_classes_count,Base_Ma.oracle_classes_count,Base_Ma.matched_attributes_count,Base_Ma.generated_attributes_count,Base_Ma.oracle_attributes_count)
                print(f'-{c}/{cycle}  {name} Baseline Matching have done!')
                print(f'{name},{cycle},{base_score[0]:4f},{base_score[1]:4f},{base_score[2]:4f},{base_score[3]:4f},{base_score[4]:4f},{base_score[5]:4f},{Base_Ma.matched_classes_count},{Base_Ma.generated_classes_count},{Base_Ma.oracle_classes_count},{Base_Ma.matched_attributes_count},{Base_Ma.generated_attributes_count},{Base_Ma.oracle_attributes_count}',file=bs)

                for j in range(0,6):
                    sum_test_result[j]+=pre_score[j]
                    sum_baseline_result[j]+=base_score[j]
                
                if c % 10 ==0:
                    # 计算中间次数的均值
                    ave_test_result = [sum_test_result[j] / c for j in range(6)]
                    ave_base_result = [sum_baseline_result[j] / c for j in range(6)]
                    # print(f'system_name,前{c}次所有测试均值,Ave_Test_Class_precision,\tAve_class_recall,\tAve_class_f1,\tAve_attribute_precision,\tAve_attribute_recall,\tAve_attribute_f1,',file=s)
                    print(f'{name},{c} average_score,{ave_test_result[0]:4f},{ave_test_result[1]:4f},{ave_test_result[2]:4f},{ave_test_result[3]:4f},{ave_test_result[4]:4f},{ave_test_result[5]:4f}',file=ps)
                    print(f'{name},{c} average_score,{ave_base_result[0]:4f},{ave_base_result[1]:4f},{ave_base_result[2]:4f},{ave_base_result[3]:4f},{ave_base_result[4]:4f},{ave_base_result[5]:4f}',file=bs)
            
            for row in log:
                if row:
                    writer.writerow([row])
        
        ave_test_result = [sum_test_result[j] / cycle for j in range(6)]
        ave_base_result = [sum_baseline_result[j] / cycle for j in range(6)]
        # 留存baseline数据到 bs 文件，prediction(test)数据到ps文件
        print(f'{name} ,test_average_score,{ave_test_result[0]:4f},{ave_test_result[1]:4f},{ave_test_result[2]:4f},{ave_test_result[3]:4f},{ave_test_result[4]:4f},{ave_test_result[5]:4f}',file=ps)
        print(f'{name} ,baseline_average_score,{ave_base_result[0]:4f},{ave_base_result[1]:4f},{ave_base_result[2]:4f},{ave_base_result[3]:4f},{ave_base_result[4]:4f},{ave_base_result[5]:4f}',file=bs)
        # 最终的测试均值  
        print(f'{name},test_final_average_score,{ave_test_result[0]:4f},{ave_test_result[1]:4f},{ave_test_result[2]:4f},{ave_test_result[3]:4f},{ave_test_result[4]:4f},{ave_test_result[5]:4f}',file=a)
        print(f'{name},base_final_average_score,{ave_base_result[0]:4f},{ave_base_result[1]:4f},{ave_base_result[2]:4f},{ave_base_result[3]:4f},{ave_base_result[4]:4f},{ave_base_result[5]:4f}',file=a)
    
        
        
        print(f'-{name}: All round have done!')
        time.sleep(5)
        


    bs.close()
    ps.close()
    a.close()
    print('Finish!')


