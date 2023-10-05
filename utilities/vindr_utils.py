import torch
import pandas as pd
import torch.nn as nn
import numpy as np


"""# Data Preparation"""

def get_df_vindr(root,pathology,cross_eval=False):
    
    train_df=pd.read_csv(root+'/image_labels_train.csv')
    valid_df=pd.read_csv(root+'/image_labels_test.csv')
    
    train_df=train_df.loc[(train_df[pathology]==1)|((train_df[pathology]==0) & (train_df['No finding']==1))]
    valid_df=valid_df.loc[(valid_df[pathology]==1)|((valid_df[pathology]==0) & (valid_df['No finding']==1))]
    train_df=train_df.drop_duplicates(subset='image_id')
    valid_df=valid_df.drop_duplicates(subset='image_id')
    #Handle the partitioning better for the sake of evaluating in the entire dataset
    test=['test' for x in range(len(valid_df))]
    valid_df=valid_df.assign(partition=test)
    train=['train' for x in range(len(train_df))]
    train_df=train_df.assign(partition=train)
    eval_df=pd.concat([train_df,valid_df],ignore_index=True)
    
    if cross_eval:
        return eval_df
    else:
        return train_df,valid_df