import torch
from torch import nn
from torch.nn import functional as F
import math
from dataloader_mm import PatientDataset
import torch.nn.utils.rnn as rnn_utils
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from collections import deque
import torch.optim as optim
from sklearn import metrics
from transformers import T5Tokenizer,AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import requests
from model_PromptMedTS_DD import PromptMedTS
import torch
import json
import dill
import copy
import time
start = time.time()
SEED = 3407 #gpu23 model 2

torch.manual_seed(SEED)
import warnings
warnings.filterwarnings('ignore')
os.environ['CUDA_VISIBLE_DEVICES']="0,1"


max_length = 500
BATCH_SIZE = 6
accumulation_step = 20
Test_batch_size = 16
pretrained = True
Freeze = False
SV_WEIGHTS = True
evaluation = True
disease_all = True
Freeze_t5coder = True
Freeze_TST = False
include_icd = False
logs = True
patch_len = 8
num_patch = 125
stride = 8
num_query_tokens = 32
if disease_all:
    Best_F1 = 0.27
else:
    Best_F1 = 0.61

if disease_all:
   save_dir= "xxx" 
else:
    save_dir= "xxx"
save_name = f"xxx"
log_file_name = f'xxx.txt'

device1 = "cuda:1" 
device1 = torch.device(device1)
device2 = "cuda:1"
device2 = torch.device(device2)
start_epoch = 0

if disease_all:
    ts_encoder_weight_dir = "xx.pth"
    prompt_encoder_weight_dir = "xxx.pth"
    t5_encoder_weight_dir = "xxx.pth"

if disease_all:
    target_diagnosis_name_list = list(json.loads( open("/home/comp/cssniu/promptt5/label_ccs_icd_d.json",'r').read()).keys()) + list(json.loads( open("/home/comp/cssniu/promptt5/label_ccs_icd_p.json",'r').read()).keys())

if evaluation:
    pretrained = False
    SV_WEIGHTS = False
    Logging = False
    weight_dir = "/home/comp/cssniu/promptt5/mmllama/weights/mimic/mmt5_stg2_0131_nofreezets_icd_ws_3407_gpu19_epoch_16_loss_0.0636_f1_micro_0.2902_f1_macro_0.2077.pth"

tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")
tokenizer_bert = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT",do_lower_case=True, local_files_only=True)


def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    data_length = [sq[0].shape[0] for sq in data]
    input_x = [i[0].tolist() for i in data]
    y = [i[1] for i in data]
    input_x = rnn_utils.pad_sequence([torch.from_numpy(np.array(x)) for x in input_x],batch_first = True, padding_value=0)

    text = [i[2] for i in data]
    label_list = [i[3] for i in data]

    return input_x,y,text,label_list


def fit(epoch,model,dataloader,optimizer,scheduler,flag='train'):
    global Best_F1,Best_Roc,patch_len,num_patch,stride
   
    if flag == 'train':
        device = device1
        model.train()

    else:
        device = device2
        model.eval()


    batch_loss_list = []

    y_list = []
    pred_list_f1 = []
    pred_list_roc = []

    model = model.to(device)
   

    for i,(lab_x,labels,text_list,label_list) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()

        label = torch.tensor(np.array(label_list)).to(torch.float32).to(device)
        if flag == "train":
            with torch.set_grad_enabled(True):
                lab_x = torch.tensor(lab_x).to(torch.float32).to(device)
                lab_x = lab_x.view(lab_x.shape[0],num_patch,lab_x.shape[-1],patch_len)
                text_input_t5 = tokenizer_t5(text_list, return_tensors="pt",padding=True).to(device)
                label_input = tokenizer_t5(labels, return_tensors="pt",padding=True).to(device)
                loss,mm_input = model(lab_x,label_input,text_input_t5)
                loss.backward(retain_graph=True)
                optimizer.step()

        else:
            with torch.no_grad():
                lab_x = torch.tensor(lab_x).to(torch.float32).to(device)
                lab_x = lab_x.view(lab_x.shape[0],num_patch,lab_x.shape[-1],patch_len)
                text_input_t5 = tokenizer_t5(text_list, return_tensors="pt",padding=True).to(device)
                label_input = tokenizer_t5(labels, return_tensors="pt",padding=True).to(device)
                loss,mm_input = model(lab_x,label_input,text_input_t5)
                output_sequences = model.t5_decoder.generate(
                    inputs_embeds = mm_input,
                    num_beams = 2,
                    max_length = 500,
                    temperature = 0.8,
                    num_return_sequences = 1,
                    # do_sample=False,
                    # length_penalty=-1,

                )  
                pred_labels = tokenizer_t5.batch_decode(output_sequences, skip_special_tokens=True)
                # print(pred_labels)
                pred = []
                for pred_label in pred_labels:
                    s_pred = [0]*len(target_diagnosis_name_list)
                    for i,d in enumerate(target_diagnosis_name_list):  
                        if d in pred_label:
                            s_pred[i] = 1  
                    pred.append(s_pred) 

                pred = np.array(pred)   
                # print(pred.shape)
                y = np.array(label.cpu().data.tolist())

                y_list.append(y)
                pred_list_f1.append(pred)
                batch_loss_list.append( loss.cpu().data )  


        y_list = np.vstack(y_list)
        pred_list_f1 = np.vstack(pred_list_f1)
        acc = metrics.accuracy_score(y_list,pred_list_f1)
        precision_micro = metrics.precision_score(y_list,pred_list_f1,average='micro')
        recall_micro =  metrics.recall_score(y_list,pred_list_f1,average='micro')
        precision_macro = metrics.precision_score(y_list,pred_list_f1,average='macro')
        recall_macro =  metrics.recall_score(y_list,pred_list_f1,average='macro')

        f1_micro = metrics.f1_score(y_list,pred_list_f1,average="micro")
        f1_macro = metrics.f1_score(y_list,pred_list_f1,average="macro")
        total_loss = sum(batch_loss_list) / len(batch_loss_list)
        end = time.time()
        running_time = end - start
        print("PHASE: {} EPOCH : {} | Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} | ACC: {} | Total LOSS  : {}  ".format(flag,epoch + 1, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,acc,total_loss))

        if logs:
            with open(f'{log_file_name}', 'a+') as log_file:
                log_file.write("PHASE: {} EPOCH : {} | Running time: {} |  Micro Precision : {} | Macro Precision : {} | Micro Recall : {} | Macro Recall : {} | Micro F1 : {} |  Macro F1 : {} |  ACC: {} | Total LOSS  : {}  ".format(flag,epoch + 1, running_time, precision_micro,precision_macro,recall_micro,recall_macro, f1_micro,f1_macro,acc,total_loss)+'\n')
                log_file.close()
        if SV_WEIGHTS:
            if f1_micro > Best_F1:
                Best_F1 = f1_micro
                PATH=f"xxx.pth"
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, PATH)

    
if __name__ == '__main__':

    train_dataset = PatientDataset(f'xx', include_icd = include_icd, disease_all = disease_all, flag="train")
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn,shuffle = True,drop_last = True)
    test_dataset = PatientDataset(f'xx', include_icd = include_icd, disease_all = disease_all,flag="test")
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=Test_batch_size, collate_fn=collate_fn,shuffle = True,drop_last = True)
    print(train_dataset.__len__())
    print(test_dataset.__len__())
    model = PromptMedTS(Freeze_t5coder = Freeze_t5coder, Freeze_TST = Freeze_TST, num_query_tokens = num_query_tokens)  # doctest: +IGNORE_RESULT
    
    if pretrained:
        t5_wights = {}
        for key, param in torch.load(t5_encoder_weight_dir,map_location=torch.device(device2)).items():
            if "t5_decoder" in key:
                t5_wights[key.replace("t5_decoder.","")] = param


        model.load_state_dict(torch.load(prompt_encoder_weight_dir,map_location=torch.device(device2)), strict=False)
        print("loading prompt_decoder weight: ",prompt_encoder_weight_dir)
        model.t5_decoder.load_state_dict(t5_wights, strict=True)
        print("loading t5_decoder weight: ",t5_encoder_weight_dir)
        model.load_state_dict(torch.load(ts_encoder_weight_dir,map_location=torch.device(device2)), strict=False )
        print("loading ts_encoder weight: ",ts_encoder_weight_dir)
    ### freeze parameters ####
    optimizer = AdamW(model.parameters(True), lr=2e-5, eps = 1e-8, weight_decay = 0.05)

    len_dataset = train_dataset.__len__()
    total_steps = (len_dataset // BATCH_SIZE) * 100 if len_dataset % BATCH_SIZE == 0 else (len_dataset // BATCH_SIZE + 1) * num_epochs 
    warm_up_ratio = 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
    if evaluation:
        model.load_state_dict(torch.load(weight_dir,map_location=torch.device(device2)), strict=True)
        print("loading weight: ",weight_dir)
        fit(1,model,testloader,optimizer,scheduler,flag='test')
     
    else:
        for epoch in range(start_epoch,num_epochs):

            fit(epoch,model,trainloader,optimizer,scheduler,flag='train')
            fit(epoch,model,testloader,optimizer,scheduler,flag='test')


            