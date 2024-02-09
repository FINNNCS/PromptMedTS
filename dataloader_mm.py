import torch
import numpy as np
import os 
import pickle
import pandas as pd
from collections import deque,Counter
from scipy import stats
import torch.nn.utils.rnn as rnn_utils
from torch.nn.utils.rnn import pad_sequence
import re
from transformers import T5Tokenizer
from tqdm import tqdm
from nltk.corpus import stopwords
import random
from datetime import datetime
from collections import defaultdict
import json
import inflect

SEED = 2019
torch.manual_seed(SEED)

tokenizer = T5Tokenizer.from_pretrained("t5-small")
p = inflect.engine()

class PatientDataset(object):
    def __init__(self, data_dir,cat_feature = False,include_icd = False,disease_all = False,flag="train",):
        self.data_dir = data_dir
        self.include_icd = include_icd
        self.flag = flag
        self.disease_all = disease_all
        self.cat_feature = cat_feature
        self.text_dir = '/home/comp/cssniu/promptt5/dataset/brief_course/'
        self.numeric_dir = '/home/comp/cssniu/promptt5/dataset/alldata/all/'
        self.icdd_parent_file = json.loads( open("/home/comp/cssniu/promptt5/ccs_icd_diag.json",'r').read())
        self.icdp_parent_file = json.loads( open("/home/comp/cssniu/promptt5/ccs_icd_proce.json",'r').read())
        self.icd_name_file = json.loads( open("/home/comp/cssniu/promptt5/label_icd_ccs.json",'r').read())
        self.css_icd_d_file = json.loads( open("/home/comp/cssniu/promptt5/label_ccs_icd_d.json",'r').read())
        self.css_icd_p_file = json.loads( open("/home/comp/cssniu/promptt5/label_ccs_icd_p.json",'r').read())
        self.css_target = json.loads( open("/home/comp/cssniu/promptt5/label_ccs_target.json",'r').read())

        self.diagnosis_codes = pd.read_csv("/home/comp/cssniu/mimiciii/mimic-iii-clinical-database-1.4/DIAGNOSES_ICD.csv")
        self.procedure_codes = pd.read_csv("/home/comp/cssniu/mimiciii/mimic-iii-clinical-database-1.4/PROCEDURES_ICD.csv")
        self.diagnosis_name = pd.read_csv("/home/comp/cssniu/mimiciii/mimic-iii-clinical-database-1.4/D_ICD_DIAGNOSES.csv")
        self.procedure_name = pd.read_csv("/home/comp/cssniu/mimiciii/mimic-iii-clinical-database-1.4/D_ICD_PROCEDURES.csv")
        self.stopword = list(pd.read_csv('/home/comp/cssniu/promptt5/stopwods.csv').values.squeeze())

        self.low = [2.80000000e+01, -7.50000000e-02,  4.30000000e+01, 4.00000000e+01,
                    4.10000000e+01,  9.00000000e+01,  5.50000000e+00,  6.15000000e+01,  
                    3.50000000e+01,  3.12996266e+01, 7.14500000e+00] 
        self.up = [  92.,           0.685,         187.,         128.,   
                    113.,         106.,          33.5,        177.5,         
                    38.55555556, 127.94021917,   7.585]   
        self.interpolation = [  59.0,           0.21,         128.0,         86.0,   
            77.0,         98.0,          19.0,        118.0,         
            36.6, 81.0,   7.4]
        self.patient_list = os.listdir(os.path.join(f'{data_dir}',flag))        
        self.max_length = 1000
        self.feature_list = [
        'Diastolic blood pressure',
        'Fraction inspired oxygen', 
        'Glucose', 
        'Heart Rate', 
        'Mean blood pressure', 
        'Oxygen saturation', 
        'Respiratory rate',
        'Systolic blood pressure', 
        'Temperature', 
        'Weight', 
        'pH']
        self.label_list = ["Acute and unspecified renal failure",
        "Acute cerebrovascular disease",
        "Acute myocardial infarction",
        "Complications of surgical procedures or medical care",
        "Fluid and electrolyte disorders",
        "Gastrointestinal hemorrhage",
        "Other lower respiratory disease",
        "Other upper respiratory disease",
        "Pleurisy; pneumothorax; pulmonary collapse",
        "Pneumonia (except that caused by tuberculosis or sexually transmitted disease)",
        "Respiratory failure; insufficiency; arrest (adult)",
        "Septicemia (except in labor)",
        "Shock",
        "Chronic kidney disease",
        "Chronic obstructive pulmonary disease and bronchiectasis",
        "Coronary atherosclerosis and other heart disease",
        "Diabetes mellitus without complication",
        "Disorders of lipid metabolism",
        "Essential hypertension",
        "Hypertension with complications and secondary hypertension",
        "Cardiac dysrhythmias",
        "Conduction disorders",
        "Congestive heart failure; nonhypertensive",
        "Diabetes mellitus with complications",
        "Other liver diseases",
        ]

        self.target_label_list = ["acute and unspecified renal failure",
        "acute cerebrovascular disease",
        "acute myocardial infarction",
        "complications of surgical procedures or medical care",
        "fluid and electrolyte disorders",
        "gastrointestinal hemorrhage",
        "other lower respiratory disease",
        "other upper respiratory disease",
        "pleurisy; pneumothorax; pulmonary collapse",
        "pneumonia",
        "respiratory failure; insufficiency; arrest",
        "septicemia",
        "shock",
        "chronic kidney disease",
        "chronic obstructive pulmonary disease and bronchiectasis",
        "coronary atherosclerosis and other heart disease",
        "diabetes mellitus without complication",
        "ddisorders of lipid metabolism",
        "essential hypertension",
        "hypertension with complications and secondary hypertension",
        "cardiac dysrhythmias",
        "conduction disorders",
        "congestive heart failure; nonhypertensive",
        "diabetes mellitus with complications",
        "other liver diseases",
        ]

        # self.target_diagnosis_list = ['49121', '51881', '5849', 'V4581', 'V1582', '0389', '78552', '5856', '4271', '40391', '7100', '99592', '42731', '28521', '2768', '7907', '99591', '00845', '5990', '2767']
        # self.target_procedure_list = [9671, 9604, 966, 3893, 3995, 3895, 3491, 4513, 5491, 3615, 3961, 40, 8872, 9390, 9904, 9907, 3891, 3722, 8856]
        self.target_diagnosis_list = ['49121', '51881', '5849', 'V4581', 'V1582', '0389', '78552', '5856', '4271', '40391']
        self.target_procedure_list = [9671, 9604, 966, 3893, 3995, 3895, 3491, 4513, 5491, 3615]
        self.origin_diagnosis_name_list = ['Obs chr bronc w(ac) exac', 'Acute respiratry failure', 'Acute kidney failure NOS', 'Aortocoronary bypass', 'History of tobacco use', 'Septicemia NOS', 'Septic shock', 'End stage renal disease', 'Parox ventric tachycard', 'Hyp kid NOS w cr kid V', 'Syst lupus erythematosus', 'Severe sepsis', 'Atrial fibrillation', 'Anemia in chr kidney dis', 'Hypopotassemia', 'Bacteremia', 'Sepsis', 'Int inf clstrdium dfcile', 'Urin tract infection NOS', 'Hyperpotassemia', 'Pleural effusion NOS', 'CHF NOS', 'Esophageal reflux', 'Alcohol cirrhosis liver', 'Chr airway obstruct NEC', 'Depressive disorder NEC', 'Thrombocytopenia NOS', 'Anemia NOS', 'Gout NOS', 'Cor ath unsp vsl ntv/gft', 'Subendo infarct, initial', 'Crnry athrscl natve vssl', 'Hypertension NOS', 'Hyperlipidemia NEC/NOS', 'Hy kid NOS w cr kid I-IV', 'Status-post ptca', 'Ac on chr syst hrt fail', 'Ascites NEC', 'Hyposmolality', 'Chronic kidney dis NOS', 'Chr pulmon heart dis NEC', 'Status cardiac pacemaker', 'DMII wo cmp nt st uncntr', 'Long-term use of insulin', 'Asthma NOS', 'Pure hypercholesterolem', 'Aortic valve disorder', 'Hypothyroidism NOS', 'Long-term use anticoagul', 'Cardiac dysrhythmias NEC', 'Anxiety state NOS', 'Hypoxemia', 'Ac posthemorrhag anemia', 'Chr diastolic hrt fail', 'Iron defic anemia NOS', 'Chr systolic hrt failure', 'Mitral valve disorder', 'Ac on chr diast hrt fail', 'Anemia-other chronic dis', 'Gastrointest hemorr NOS', 'Hyperosmolality', 'Diarrhea', 'Tobacco use disorder', 'Pneumonia, organism NOS', 'Acidosis', 'Long-term use steroids', 'Abn react-procedure NEC', 'Ac kidny fail, tubr necr', 'Morbid obesity', 'Epilep NOS w/o intr epil', 'Hx-ven thrombosis/embols', 'Hx TIA/stroke w/o resid', 'Chrnc hpt C wo hpat coma', 'Delirium d/t other cond', 'Osteoporosis NOS', 'Dehydration', 'Pressure ulcer, low back', 'Convulsions NEC', 'Dysthymic disorder', 'Surg compl-heart', 'Food/vomit pneumonitis', 'BPH w/o urinary obs/LUTS', 'Obstructive sleep apnea', 'Hypotension NOS', 'Hypovolemia', 'Cirrhosis of liver NOS', 'Abn react-surg proc NEC', 'Obesity NOS', 'Hx of past noncompliance', 'DMII neuro nt st uncntrl', 'Neuropathy in diabetes', 'DMII renl nt st uncntrld', 'Nephritis NOS in oth dis', 'Pulmonary collapse', 'Protein-cal malnutr NOS', 'React-oth vasc dev/graft', 'Prim cardiomyopathy NEC', 'Periph vascular dis NOS', 'Iatrogenc hypotnsion NEC', 'Renal dialysis status', 'Old myocardial infarct', 'Do not resusctate status', 'Chronic pain NEC', 'Atrial flutter', 'DMI neuro uncntrld', 'Gastroparesis', 'DMI ketoacd uncontrold', 'Diabetic retinopathy NOS', 'Mal hyp kid w cr kid V', 'DMI neuro nt st uncntrld']
        self.origin_procedure_name_list = ['Cont inv mec ven <96 hrs', 'Insert endotracheal tube', 'Entral infus nutrit sub', 'Venous cath NEC', 'Hemodialysis', 'Ven cath renal dialysis', 'Thoracentesis', 'Sm bowel endoscopy NEC', 'Percu abdominal drainage', '1 int mam-cor art bypass', 'Extracorporeal circulat', 'Procedure-one vessel', 'Dx ultrasound-heart', 'Non-invasive mech vent', 'Packed cell transfusion', 'Serum transfusion NEC', 'Arterial catheterization', 'Left heart cardiac cath', 'Coronar arteriogr-2 cath', 'Cont inv mec ven 96+ hrs', 'CV cath plcmt w guidance', 'Closed bronchial biopsy', 'Spinal tap', 'Rt/left heart card cath', 'Lingual thyroid excision', 'Temporary tracheostomy']
       
        self.target_diagnosis_name_list = [
        'obstructive chronic bronchitis', 
        'acute respiratry failure', 
        'acute kidney failure', 
        'aortocoronary bypass', 
        'tobacco use', 
        'septicemia', 
        'septic shock', 
        'end stage renal', 
        'parox ventric tachycard', 
        'hypertensive chronic kidney']

        self.target_procedure_name_list = [
        'invasive mechanical ventilation', 
        'endotracheal tube', 
        'entral infusion nutrit', 
        'venous cath', 
        'hemodialysis', 
        'renal dialysis', 
        'thoracentesis', 
        'small bowel endoscopy', 
        'percu abdominal drainage', 
        'mammary coronary artery bypass'] 

      

    def data_processing(self,data):

        return ''.join([i.lower() for i in data if not i.isdigit()])
    def padding_text(self,vec):
        input_ids = vec['input_ids']
        attention_mask = vec['attention_mask']
        padding_input_ids = torch.ones((input_ids.shape[0],self.max_length-input_ids.shape[1]),dtype = int).to(self.device)
        padding_attention_mask = torch.zeros((attention_mask.shape[0],self.max_length-attention_mask.shape[1]),dtype = int).to(self.device)
        input_ids_pad = torch.cat([input_ids,padding_input_ids],dim=-1)
        attention_mask_pad = torch.cat([attention_mask,padding_attention_mask],dim=-1)
        vec = {'input_ids': input_ids_pad,
        'attention_mask': attention_mask_pad}
        return vec
    def sort_key(self,text):
        temp = []
        id_ = int(re.split(r'(\d+)', text.split("_")[-1])[1])
        temp.append(id_)

        return temp
    def rm_stop_words(self,text):
            tmp = text.split(" ")
            for t in self.stopword:
                while True:
                    if t in tmp:
                        tmp.remove(t)
                    else:
                        break
            text = ' '.join(tmp)
            # print(len(text))
            return text
    def __getitem__(self, idx):
        patient_file = self.patient_list[idx]
        breif_course_list = []
        event_lab_list = []
        label_list = []
        label_name_list = []
        numeric_data = None
        event_exist = True
        lab_exist = True
        text_df = pd.read_csv(self.text_dir+"_".join(patient_file.split("_")[:2])+".csv").values
        breif_course = text_df[:,1:2].tolist()
        breif_course = [str(i[0]) for i in breif_course if not str(i[0]).isdigit()]
        text = ' '.join(breif_course)
        text = self.rm_stop_words(text)

        numeric_data_file =  self.numeric_dir + patient_file.split("_")[0] + "_" + patient_file.split("_")[2].replace("eposide","episode").strip(".csv") + "_timeseries.csv"
        lab_dic = defaultdict(list)

        lab_description = []

        if not os.path.exists(numeric_data_file):
            numeric_data = np.array([self.interpolation]*24)
            for l in range(numeric_data.shape[-1]):
                descp = f"{self.feature_list[l]} is normal all the time"
                lab_description.append(descp.lower())

        else:
            numeric_data = pd.read_csv(numeric_data_file)[self.feature_list].values
            for l in range(numeric_data.shape[-1]):
                for s in np.array(numeric_data[:,l]):
                    if s <= self.low[l]:
                        lab_dic[l].append("low")
                    elif s > self.up[l]:
                        lab_dic[l].append("high")
                    else:
                        lab_dic[l].append("normal")
            # print()
            for k in lab_dic.keys():
                risk_types = set(lab_dic[k])
                for r in risk_types:
                    # print(r)
                    if r != "normal":
                        length_r = len(np.where(np.array(lab_dic[k])==r)[0])
                        descp = self.feature_list[k] + f" is {r}er than normal {p.number_to_words(length_r)} times"
                        # pass
                    else:
                        descp = self.feature_list[k] + f" is normal all the time"
                    lab_description.append(descp.lower())

     

        lab_description = ','.join(lab_description)

        if self.disease_all:

            text = text + ". The diagnosis are <extra_id_2>, the procedures are <extra_id_3>"

            subj_id = patient_file.split("_")[0]

            diagnosis_code = self.diagnosis_codes.loc[ self.diagnosis_codes["SUBJECT_ID"]==int(subj_id)]["ICD9_CODE"].values
            procedure_code = self.procedure_codes.loc[ self.procedure_codes["SUBJECT_ID"]==int(subj_id)]["ICD9_CODE"].values
            diagnosis_codes = diagnosis_code[~pd.isnull(diagnosis_code)]
            procedure_codes = procedure_code[~pd.isnull(procedure_code)]
            diagnosis_code = [i for i in diagnosis_codes if i in list(self.icd_name_file.keys())]
            procedure_code = [i for i in procedure_codes if str(i) in list(self.icd_name_file.keys())]



            diagnosis_names = [self.icd_name_file[i] for i in diagnosis_code]
            procedure_names = [self.icd_name_file[str(i)] for i in procedure_code]

            diagnosis_labels = [0]*len(self.css_icd_d_file.keys())
            procedure_labels = [0]*len(self.css_icd_p_file.keys())
            
            diagnosis_names_tobepred = []
            procedure_names_tobepred = []
            for i,d in enumerate(self.css_icd_d_file.keys()):
                if d in diagnosis_names:
                    diagnosis_labels[i] = 1
                    if self.include_icd:
                        diagnosis_names_tobepred.append(list(self.css_icd_d_file.keys())[i]+"--"+list(self.css_icd_d_file.values())[i])
                    else:
                        diagnosis_names_tobepred.append(list(self.css_icd_d_file.keys())[i])

            for i,d in enumerate(self.css_icd_p_file.keys()):
                if d in procedure_names:
                    procedure_labels[i] = 1
                    if self.include_icd:
                        procedure_names_tobepred.append(list(self.css_icd_p_file.keys())[i]+"--"+list(self.css_icd_p_file.values())[i])
                    else:
                        procedure_names_tobepred.append(list(self.css_icd_p_file.keys())[i])



            label_names = "<extra_id_2> "+','.join(diagnosis_names_tobepred) + ",<extra_id_3> " +','.join(procedure_names_tobepred) + " <extra_id_4>" 

            label = diagnosis_labels + procedure_labels

        else:
            label = pd.read_csv(os.path.join(self.data_dir,self.flag,patient_file))[self.label_list].values[:1,:][0]
            label_index = np.where(label == 1)[0]
            text = text + ". The diagnosis are <extra_id_2>"
            label_names = [self.label_list[i] for i in label_index]
            label_names =  "<extra_id_2>"+','.join(label_names)+ "<extra_id_3>" 

        if len(numeric_data) < self.max_length:
            numeric_data = np.concatenate((numeric_data, np.repeat(np.expand_dims(numeric_data[-1,:], axis=0),1000-len(numeric_data),axis = 0) ), axis=0)

        return numeric_data,label_names,text,label,lab_description


    def __len__(self):
        return len(self.patient_list)


