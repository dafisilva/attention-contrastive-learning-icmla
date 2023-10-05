import torch
import numpy as np
import matplotlib.pyplot as plt 
import torchvision
from torch.utils.data import DataLoader as DataLoader
from torchvision import transforms as transforms
from tqdm import tqdm
import torch.nn as nn
import torchvision.utils as vutils
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import shutil
import torchmetrics
import wandb
#from config.config import *
from datasets.dataset_brax import *
from models.models_brax import *
from utilities.brax_utils import *

import gc



import argparse
parser=argparse.ArgumentParser()

parser.add_argument("fold",type=int,default=1)

args=parser.parse_args()

FOLD=args.fold

pathology='Atelectasis'

"""GET DATAFRAME FOR BRAX DATASET"""
np.random.seed(0)
torch.manual_seed(0)

train_df,test_df=get_df_brax('csv_path',pathology)

xray_feat_df=pd.read_csv('cluster_label_csv')

train_df['PngPath']=train_df['PngPath'].str.replace('images/','')
test_df['PngPath']=test_df['PngPath'].str.replace('images/','')

train_df=pd.merge(train_df,xray_feat_df,on=['PngPath'])
test_df=pd.merge(test_df,xray_feat_df,on=['PngPath'])

target=train_df[pathology].to_numpy().astype(int)
skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)

transf_v=transforms.Compose([transforms.Resize((256,256)),
                             transforms.ToTensor()])

transf=transforms.Compose([transforms.Resize((256,256)),                         
                               transforms.RandomResizedCrop(size=256,scale=(0.7,1.0)),
                               transforms.RandomAffine(degrees=5,translate=(0.1,0.1)),
                               #transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.2),
                               transforms.ToTensor()])


data_loader_fold=[]
root_interpol='sampled_brax_dataset_path'
root_orig='original_brax_dataset_path'
data_dict=[{'train':train_df.iloc[train_idx],'valid':train_df.iloc[valid_idx]} for train_idx,valid_idx in skf.split(train_df['PngPath'],target)]

for fold in range(len(data_dict)):
    
    trainset=Dataset_BRAX_MultiHead_xray_interpol(data_dict[fold]['train'],root_orig,root_interpol,transf,pathology)
    validset=Dataset_BRAX_MultiHead_xray_interpol(data_dict[fold]['valid'],root_orig,root_interpol,transf_v,pathology)
    
    target_fold=data_dict[fold]['train'][pathology].to_numpy().astype(int)
    class_sample_count = np.array([len(np.where(target_fold == t)[0]) for t in np.unique(target_fold)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target_fold])
    samples_weight = torch.from_numpy(samples_weight)
    sampler_train = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
    
    
    
    trainloader=DataLoader(trainset,batch_size=16,sampler=sampler_train,num_workers=8,pin_memory=True)
    validloader=DataLoader(validset,batch_size=16,num_workers=8,pin_memory=True,drop_last=True)
    
    data_loader_fold.append((trainloader,validloader))


trainloader,validloader=data_loader_fold[FOLD]


print('DATA LOADED!')

"""# Data and Model init"""



min_v_auroc_x=0

model=Tiny_MH_EP_AM_interpol()


"""# Training Loop"""
device= 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)
epochs=50
lr=1e-4
criterion_dis=nn.BCEWithLogitsLoss()  #Disease Classification
criterion_x=nn.CrossEntropyLoss()    #xray features Classification
cos=nn.CosineEmbeddingLoss()
mse=nn.MSELoss()
PATH='model_checkpoint_path'



model=model.to(device)


freeze(model,['linear_dis','linear_x'])
optim_eq=torch.optim.Adam([param for param in model.parameters() if param.requires_grad==True],lr=1e-4)
unfreeze(model,['linear_dis','linear_x'])
optim_total=torch.optim.Adam([param for param in model.parameters() if param.requires_grad==True],lr=1e-4)

acc_d=torchmetrics.classification.BinaryAccuracy(threshold=0.5).to(device) #Binary
f1_d=torchmetrics.classification.BinaryF1Score()
auroc_d=torchmetrics.classification.BinaryAUROC()
c_mat_d=torchmetrics.classification.BinaryConfusionMatrix(threshold=0.5)

acc_x=torchmetrics.classification.MulticlassAccuracy(4).to(device)
f1_x=torchmetrics.classification.MulticlassF1Score(4)
auroc_x=torchmetrics.classification.MulticlassAUROC(4)

scheduler_1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_total, 'max',verbose=True)
scheduler_2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim_eq, 'max',verbose=True)



for epoch in range(epochs):

    print(f'EPOCH {epoch}/{epochs}')

    t_loss=0
    
    t_x_loss=0
    t_d_loss=0
    
    t_x_acc=0
    t_d_acc=0

    preds_d=[]
    targets_d=[]
    
    preds_x=[]
    targets_x=[]
    
    pos=torch.Tensor([1]).to(device)
    neg=torch.Tensor([-1]).to(device)
    
    model.train()

    print('TRAINING MODEL...')

    for input_orig,input_interpol,label_d,label_x in tqdm(trainloader):

        optim_eq.zero_grad()
        optim_total.zero_grad()
        input_orig=input_orig.to(device)
        input_interpol=input_interpol.to(device)
        label_x=label_x.to(device)
        label_d=label_d.to(device)
        
        logits_d_orig,logits_x_orig,map_dis_o,map_x_o=model(input_orig,'classification')
        
        logits_d_interpol,logits_x_interpol,map_dis_i,map_x_i=model(input_interpol,'classification')
        
        label_x_orig=torch.Tensor([[3]*input_orig.shape[0]])[0].to(device)

        
        loss_d=(criterion_dis(logits_d_orig.squeeze().float(),label_d.float())+criterion_dis(logits_d_interpol.squeeze().float(),label_d.float()))/2

        loss_x=(criterion_x(logits_x_orig.squeeze(),label_x_orig.long())+criterion_x(logits_x_interpol.squeeze(),label_x.long()))/2
        attn_loss=0
        for i in range(len(map_dis_o)):
            
            attn_loss+=mse(map_dis_o[i],map_dis_i[i])/len(map_dis_o)
        
        contrast_loss=attn_loss+cos(map_x_o,map_x_i,neg)
        
        
        loss=loss_d+loss_x+contrast_loss

        loss.backward()
        
        optim_total.step()
        eq_loss=model(input_interpol,'eq')
        eq_loss.backward()
        
        optim_eq.step()
        

        t_d_acc+=acc_d(logits_d_interpol.squeeze(),label_d).item()
        t_x_acc+=acc_x(logits_x_interpol.squeeze(),label_x).item()
        
        
        t_loss+=loss.item()
        t_d_loss+=loss_d.item()
        t_x_loss+=loss_x.item()
        
        preds_d.append(logits_d_interpol.detach().cpu())
        targets_d.append(label_d.detach().cpu())
        
        preds_x.append(logits_x_interpol.detach().cpu())
        targets_x.append(label_x.detach().cpu())




    preds_d=torch.cat(preds_d)
    targets_d=torch.cat(targets_d)
    
    preds_x=torch.cat(preds_x)
    targets_x=torch.cat(targets_x)    
    
    t_f1_d=f1_d(preds_d.squeeze(),targets_d.int())
    t_auroc_d=auroc_d(preds_d.squeeze(),targets_d.int())
    
    t_f1_x=f1_x(preds_x.squeeze(),targets_x.int())
    t_auroc_x=auroc_x(preds_x.squeeze(),targets_x.int())  
      
    print(c_mat_d(preds_d.squeeze(),targets_d.int()))
    
    t_loss=t_loss/len(trainloader)
    
    t_d_acc=t_d_acc/len(trainloader)
    t_x_acc=t_x_acc/len(trainloader)

    t_d_loss=t_d_loss/len(trainloader)
    t_x_loss=t_x_loss/len(trainloader)
    


    wandb.log({"t_loss":t_loss,"t_xray_loss":t_x_loss,"t_dis_loss":t_d_loss,"t_dis_acc":t_d_acc,"t_xray_acc":t_x_acc,"t_dis_f1":t_f1_d,"t_xray_f1":t_f1_x,"t_dis_auc":t_auroc_d,"t_xray_auc":t_auroc_x})
    print('VALIDATING MODEL...')
    v_loss=0
    
    v_x_loss=0
    v_d_loss=0
    v_x_acc=0
    v_d_acc=0
    preds_d=[]
    targets_d=[]
    preds_x=[]
    targets_x=[]
    model.eval()
    with torch.no_grad():

        for _,input_interpol,label_d,label_x in tqdm(validloader):

            #input_orig=input_orig.to(device)
            input_interpol=input_interpol.to(device)
            label_x=label_x.to(device)
            label_d=label_d.to(device)
            logits_d,logits_x,_,_=model(input_interpol)
            
            loss_d=criterion_dis(logits_d.squeeze().float(),label_d.float())
            loss_x=criterion_x(logits_x.squeeze(),label_x.long())

            loss=loss_d+loss_x
        
            v_d_acc+=acc_d(logits_d.squeeze(),label_d).item()
            v_x_acc+=acc_x(logits_x.squeeze(),label_x).item()
            
            v_loss+=loss.item()
            v_d_loss+=loss_d.item()
            v_x_loss+=loss_x.item()

            preds_d.append(logits_d.detach().cpu())
            targets_d.append(label_d.detach().cpu())
            
            preds_x.append(logits_x.detach().cpu())
            targets_x.append(label_x.detach().cpu())

    v_loss=v_loss/len(validloader)
    v_d_acc=v_d_acc/len(validloader)
    v_x_acc=v_x_acc/len(validloader)
    
    v_d_loss=v_d_loss/len(validloader)
    v_x_loss=v_x_loss/len(validloader)
    


    preds_d=torch.cat(preds_d)
    targets_d=torch.cat(targets_d)
    
    preds_x=torch.cat(preds_x)
    targets_x=torch.cat(targets_x)    
    
    v_f1_d=f1_d(preds_d.squeeze(),targets_d.int())
    v_auroc_d=auroc_d(preds_d.squeeze(),targets_d.int())
    
    v_f1_x=f1_x(preds_x.squeeze(),targets_x.int())
    v_auroc_x=auroc_x(preds_x.squeeze(),targets_x.int())  
    
    print(c_mat_d(preds_d.squeeze(),targets_d.int()))

    

    wandb.log({"v_loss":v_loss,"v_xray_loss":v_x_loss,"v_dis_loss":v_d_loss,"v_dis_acc":v_d_acc,"v_xray_acc":v_x_acc,"v_dis_f1":v_f1_d,"v_xray_f1":v_f1_x,"v_dis_auc":v_auroc_d,"v_xray_auc":v_auroc_x})
    

    
    print(f'Training Loss: {t_loss}  \nTraining Dis Acc:{t_d_acc}  \nTraining Xray Acc:{t_x_acc} \nVal Loss: {v_loss}  \nVal Dis Acc: {v_d_acc} \nVal Xray Acc: {v_x_acc}')
    
    scheduler_1.step(v_auroc_d)
    scheduler_2.step(v_auroc_d)
    if v_auroc_d>=min_v_auroc_d:
      min_v_auroc_d=v_auroc_d
      print('Validation Disease AUROC increased! SAVING MODEL...')
      torch.save(model.state_dict(), os.path.join(PATH,'model_checkpoint_best_dis_auroc_fold_'+str(FOLD)+'.pth'))

