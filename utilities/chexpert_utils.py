import torch
import pandas as pd
import torch.nn as nn
import numpy as np
import os

"""# Data Preparation"""

def get_df_chex(root,pathology,cross_eval=False):
    
    train_df=pd.read_csv(os.path.join(root,'train.csv'))
    valid_df=pd.read_csv(os.path.join(root,'valid.csv'))
    
    train_df=train_df.fillna(0)
    valid_df=valid_df.fillna(0)
    
    train_df=train_df.loc[(train_df['AP/PA']=='PA') &  (train_df['Frontal/Lateral']=='Frontal') & ((train_df[pathology]==1)|((train_df[pathology]==0) & (train_df['No Finding']==1)))].reset_index()
    valid_df=valid_df.loc[(valid_df['AP/PA']=='PA') &  (valid_df['Frontal/Lateral']=='Frontal')  & ((train_df[pathology]==1)|((train_df[pathology]==0) & (train_df['No Finding']==1)))].reset_index()
    

    eval_df=pd.concat([train_df,valid_df],ignore_index=True)
    paths=[x.replace(x.split('/')[0]+'/','') for x in list(eval_df['Path'])]

    paths_t=[x.replace(x.split('/')[0]+'/','') for x in list(train_df['Path'])]
    paths_v=[x.replace(x.split('/')[0]+'/','') for x in list(valid_df['Path'])]
    
    eval_df['Path']=paths
    
    train_df['Path']=paths_t
    valid_df['Path']=paths_v
    if cross_eval:
        return eval_df
    else:
        return train_df,valid_df
    
    
    
