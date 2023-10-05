import torch
import pandas as pd
import torch.nn as nn
import numpy as np

from sklearn.model_selection import StratifiedKFold
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader


"""# Data Preparation"""


def get_df_brax(df_path,pathology,evalua=False):
    np.random.seed(0)
    
    df=pd.read_csv(df_path)

    positions=df['ViewPosition'].unique()
    number=[]
    for pos in positions:

        number.append(sum(df['ViewPosition']==pos))

    PA_df=df.loc[df['ViewPosition']=='PA']
    PA_df=PA_df.replace(to_replace=[5,6],value=[4,5])
    pa_unique_df = PA_df.drop_duplicates(subset=["PatientID"], keep=False)
    diseases=['No Finding',
        'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Lesion',
        'Lung Opacity', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis',
        'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture']

    for dis in diseases:

        num=sum(pa_unique_df[dis]==1)
        print(f'amount of cases with {dis}:{num}')

    brax_df=pa_unique_df.loc[((pa_unique_df[pathology]==1) & (pa_unique_df[pathology]!=-1))| ((pa_unique_df['No Finding']==1) & (pa_unique_df[pathology]!=-1))]
    brax_df=brax_df.fillna(0)

    positive=brax_df.loc[brax_df[pathology]==1]
    negative=brax_df.loc[brax_df[pathology]==0] #.sample(frac=0.3,random_state=0)

    train_pos=positive.sample(frac = 0.8,random_state=0)
    test_pos=positive.drop(train_pos.index)

    train_neg=negative.sample(frac = 0.8,random_state=0)
    test_neg=negative.drop(train_neg.index)

    train_df=pd.concat([train_pos,train_neg],axis=0).reset_index().sample(frac=1)
    test_df=pd.concat([test_pos,test_neg],axis=0).reset_index().sample(frac=1)
    eval_df=pd.concat([train_df,test_df],ignore_index=True)
    if evalua==True:
        return eval_df
    else:
        return train_df,test_df


    
    


"""Freeze Layers"""



    
    
    
def freeze(model,attributes_f):
    for attr in attributes_f:
        
        for i,name in getattr(model,attr).named_modules():
            if str(i)!='':
                for j,name in getattr(model,attr)[int(i)].named_parameters():
                    getattr(getattr(model,attr)[int(i)],j).requires_grad=False
                    
def unfreeze(model,attributes_f):
    for attr in attributes_f:
        
        for i,name in getattr(model,attr).named_modules():
            if str(i)!='':
                for j,name in getattr(model,attr)[int(i)].named_parameters():
                    getattr(getattr(model,attr)[int(i)],j).requires_grad=True
    


def get_loaders_brax_abl(train_df,test_df,pathology,root):
    root_interpol=root
    root_orig='/processing/d.barros/brax_orig_resized'
    transf_v=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

    transf=transforms.Compose([transforms.Resize((256,256)),                         
                                transforms.RandomResizedCrop(size=256,scale=(0.7,1.0)),
                                transforms.RandomAffine(degrees=5,translate=(0.1,0.1)),
                                #transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.2),
                                transforms.ToTensor()])

    train_df['PngPath']=train_df['PngPath'].str.replace('images/','')
    test_df['PngPath']=test_df['PngPath'].str.replace('images/','')
    target=train_df[pathology].to_numpy().astype(int)

    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)

    data_loader_fold=[]

    data_dict=[{'train':train_df.iloc[train_idx],'valid':train_df.iloc[valid_idx]} for train_idx,valid_idx in skf.split(train_df['PngPath'],target)]

    for fold in range(len(data_dict)):
        
        
        trainset=Dataset_BRAX_ablation(data_dict[fold]['train'],root_orig,root_interpol,transf,pathology)
        validset=Dataset_BRAX_ablation(data_dict[fold]['valid'],root_orig,root_interpol,transf_v,pathology)
        
        target_fold=data_dict[fold]['train'][pathology].to_numpy().astype(int)
        class_sample_count = np.array([len(np.where(target_fold == t)[0]) for t in np.unique(target_fold)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target_fold])
        samples_weight = torch.from_numpy(samples_weight)
        sampler_train = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        
        
        
        trainloader=DataLoader(trainset,batch_size=16,sampler=sampler_train,num_workers=8)
        validloader=DataLoader(validset,batch_size=16,drop_last=True,num_workers=8)
        
        data_loader_fold.append((trainloader,validloader))
        
    return data_loader_fold
