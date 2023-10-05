
import torch
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
from torch.utils.data import DataLoader as DataLoader
from torchvision import transforms as transforms
import torchvision.utils as vutils
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
import torchmetrics
import wandb
from models.models_brax import *
from utilities.mimic_utils import *
from datasets.dataset_mimic import *


import argparse
parser=argparse.ArgumentParser()

parser.add_argument("fold",type=int,default=1)

args=parser.parse_args()

FOLD=args.fold


pathology='Atelectasis'

wandb.init(
    name='Model Test: MIMIC fold ' +str(FOLD),
    

)



eval_df=get_df_mimic('csv_path',pathology,True)



print('DATA LOADED!')




"""# Data and Model init"""

root='insert_data_folder'

batch_size=8
min_loss=1e6
np.random.seed(0)
torch.manual_seed(0)

transf_neg=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])



evalset=Dataset_MIMIC(eval_df,root,transf_neg,pathology)


evalloader=DataLoader(evalset,batch_size,num_workers=12)

device= 'cuda:0' if torch.cuda.is_available() else 'cpu'



model=Tiny_MH_EP_AM_interpol()

model=model.to(device)
model.load_state_dict(torch.load('model_checkpoint_path'+str(FOLD)+'.pth'))

"""# Training Loop"""


criterion=nn.BCEWithLogitsLoss()  #Binary Classification

acc=torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(device) #Binary
f1=torchmetrics.classification.BinaryF1Score()
roc=torchmetrics.classification.BinaryROC()
auroc=torchmetrics.classification.BinaryAUROC()
sig=nn.Sigmoid()




    
print('TESTING MODEL...')
t_loss=0
t_acc=0
preds=[]
targets=[]
model.eval()


for inputs,labels in tqdm(evalloader):

    

    inputs=inputs.to(device)
    labels=labels.to(device)

    dis_logits,_,_=model(inputs)

    if len(inputs)==1:
        #dis_loss=criterion(outputs[0],labels.float())
        #eq_prob_loss=model.mse_eq_prob(out_m,len(out_m),device)
        #loss=dis_loss+eq_prob_loss
        
        loss=criterion(dis_logits[0],labels.float())
        t_acc+=acc(dis_logits[0],labels).item()
    else:
        #dis_loss=criterion(outputs.squeeze().float(),labels.float())
        #eq_prob_loss=model.mse_eq_prob(out_m,len(out_m),device)
        
        #loss=dis_loss+eq_prob_loss
        loss=criterion(dis_logits.squeeze().float(),labels.float())
        t_acc+=acc(dis_logits.squeeze().float(),labels.float()).item()

    t_loss+=loss.item()
    preds.append(dis_logits.detach().cpu())
    targets.append(labels.detach().cpu())
preds=torch.cat(preds)
targets=torch.cat(targets)

f1_score=f1(preds.squeeze(),targets.int())
auc=auroc(preds.squeeze(),targets.int())

t_loss=t_loss/len(evalloader)
t_acc=t_acc/len(evalloader)

wandb.log({'Loss':t_loss,'Accuracy':t_acc,'F1 Score':f1_score,'AUROC':auc})
