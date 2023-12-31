
"""classification.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Xu8zs9QMBHp51Pc-G7q3mxnenkHOWpgY

# Imports
"""
import gc
from sklearn.manifold import TSNE
import torch
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
from torch.utils.data import DataLoader as DataLoader
from torchvision import transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
import torchmetrics
import wandb
from sklearn.model_selection import StratifiedKFold

from datasets.dataset_brax import *
from models.models_brax import *
from utilities.brax_utils import *


import argparse
parser=argparse.ArgumentParser()

parser.add_argument("fold",type=int,default=1)

args=parser.parse_args()

FOLD=args.fold

pathology='Atelectasis'



"""GET DATAFRAME FOR BRAX DATASET"""

train_df,test_df=get_df_brax('root_for_csv',pathology)

root='brax_dataset_path'


batch_size=16

np.random.seed(0)
torch.manual_seed(0)

test_df['PngPath']=test_df['PngPath'].str.replace('images/','')
transf=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

testset=Dataset_BRAX(test_df,root,transf,pathology)
    
testloader=DataLoader(testset,batch_size=16,num_workers=12)
    

print('DATA LOADED!')



wandb.init(
    
    name='Model Test: BRAX - fold '+str(FOLD),
    config={"epochs":100,"batch_size":16,"lr":1e-5,"criterion":"BCEWithLogitsLoss"}

)


model=Tiny_MH_EP_AM_interpol()

model=model.to(device)
model.load_state_dict(torch.load('model_checkpoint_path'+str(FOLD)+'.pth'))

"""# Testing Loop"""
device= 'cuda:0' if torch.cuda.is_available() else 'cpu'


criterion=nn.BCEWithLogitsLoss() #Binary Classification

acc=torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(device) #Binary
f1=torchmetrics.classification.BinaryF1Score()
c_mat=torchmetrics.classification.BinaryConfusionMatrix(threshold=0.5)
auroc=torchmetrics.classification.BinaryAUROC()

sig=nn.Sigmoid()



    
print('TESTING MODEL...')
test_loss=0

test_acc=0
preds=[]
targets=[]
model.eval()
with torch.no_grad():
    
    for inputs,labels in tqdm(testloader):

        inputs=inputs.to(device)
        labels=labels.to(device)

        out_d,_,_=model(inputs)
        if len(inputs)==1:
            loss=criterion(sig(out_d[0]).float(),labels)
            test_acc+=acc(out_d[0],labels).item()

        else:

            loss=criterion(out_d.squeeze().float(),labels)
            test_acc+=acc(out_d.squeeze(),labels).item()
        
        test_loss+=loss.item()
        preds.append(out_d.detach().cpu())
        targets.append(labels.detach().cpu())
           
            

test_loss=test_loss/len(testloader)
test_acc=test_acc/len(testloader)

preds=torch.cat(preds)
targets=torch.cat(targets)
test_auroc=auroc(preds.squeeze(),targets.int())
test_f1=f1(preds.squeeze(),targets.int())
print(c_mat(preds.squeeze(),targets.int()))

wandb.log({"validation_loss":test_loss,"validation_accuracy":test_acc,"validation_f1":test_f1,"validation_auroc":test_auroc})




