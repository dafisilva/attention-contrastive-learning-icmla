from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader as DataLoader
from sklearn.model_selection import StratifiedKFold
from datasets.dataset_mimic import *
def get_df_mimic(df_path,pathology,cross_eval=False):
    np.random.seed(0)
    
     
    mdata_df=pd.read_csv(os.path.join(df_path,'mimic-cxr-2.0.0-metadata.csv.gz'))
    chexpert_df=pd.read_csv(os.path.join(df_path,'mimic-cxr-2.0.0-chexpert.csv.gz'))
    mimic_df=pd.merge(mdata_df,chexpert_df,on=['subject_id','study_id'])
    mimic_df=mimic_df.loc[mimic_df['ViewPosition']=='PA']
    
    mimic_df['PngPath']=[os.path.join('files',os.path.join('p'+mimic_df['subject_id'].iloc[i].astype(str)[:2],os.path.join('p'+mimic_df['subject_id'].iloc[i].astype(str),os.path.join('s'+mimic_df['study_id'].iloc[i].astype(str),mimic_df['dicom_id'].iloc[i]+'.jpg')))) for i in range(len(mimic_df))]
    mimic_df2=mimic_df.loc[(mimic_df[pathology]==1) | (mimic_df[pathology]==0)|(mimic_df['No Finding']==1)].reset_index(drop=True)
    mimic_df2[pathology]=mimic_df2[pathology].fillna(0)
    
    mimic_df3=mimic_df2.drop_duplicates(subset=['subject_id'])
    skf=StratifiedShuffleSplit(1,test_size=0.2,random_state=0)
    
    data_dict=[{'train':mimic_df3.iloc[train_idx],'test':mimic_df3.iloc[test_idx]} for train_idx,test_idx in skf.split(mimic_df3['subject_id'],mimic_df3[pathology])]
    
    train_df=data_dict[0]['train']
    test_df=data_dict[0]['test']
    
    train_full=mimic_df2.loc[mimic_df2['subject_id'].isin(train_df['subject_id'])].sample(frac=1).reset_index(drop=True)
    test_full=mimic_df2.loc[mimic_df2['subject_id'].isin(test_df['subject_id'])].sample(frac=1).reset_index(drop=True)
    
    
    if cross_eval:
        eval_df=pd.concat([train_full,test_full],ignore_index=True)
        return eval_df
    
    else:

        return train_full,test_full


def get_loaders_mimic(train_df,test_df,pathology,root):
    
    transf_v=transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

    transf=transforms.Compose([transforms.Resize((256,256)),                         
                                transforms.RandomResizedCrop(size=256,scale=(0.7,1.0)),
                                transforms.RandomAffine(degrees=5,translate=(0.1,0.1)),
                                #transforms.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.2),
                                transforms.ToTensor()])


    target=train_df[pathology].to_numpy().astype(int)
    skf=StratifiedKFold(n_splits=5,random_state=0,shuffle=True)

    data_loader_fold=[]


    train_df_nd=train_df.drop_duplicates(subset=['subject_id'])
    print('Length of drop duplicates',len(train_df_nd))
    test_df_nd=test_df.drop_duplicates(subset=['subject_id'])


    data_dict=[{'train':train_df_nd.iloc[train_idx],'valid':train_df_nd.iloc[valid_idx]} for train_idx,valid_idx in skf.split(train_df_nd['subject_id'],train_df_nd[pathology])]

    for fold in range(len(data_dict)):
        
        
        train_df_final=train_df.loc[train_df['subject_id'].isin(data_dict[fold]['train']['subject_id'])]
        
        print('Length of original dataframe',len(train_df))
        valid_df_final=train_df.loc[train_df['subject_id'].isin(data_dict[fold]['valid']['subject_id'])]
        print('Length of final dataframe',len(train_df_final)+len(valid_df_final))
        
        print(train_df_final[pathology].value_counts(True))
        print(valid_df_final[pathology].value_counts(True))
        trainset=Dataset_MIMIC(train_df_final,root,transf,pathology)
        validset=Dataset_MIMIC(valid_df_final,root,transf_v,pathology)
        
        target_fold=data_dict[fold]['train'][pathology].to_numpy().astype(int)
        class_sample_count = np.array([len(np.where(target_fold == t)[0]) for t in np.unique(target_fold)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target_fold])
        samples_weight = torch.from_numpy(samples_weight)
        sampler_train = torch.utils.data.sampler.WeightedRandomSampler(samples_weight, len(samples_weight))
        
        
        
        trainloader=DataLoader(trainset,batch_size=16,sampler=sampler_train)
        validloader=DataLoader(validset,batch_size=16)
        
        data_loader_fold.append((trainloader,validloader))
        
    return data_loader_fold