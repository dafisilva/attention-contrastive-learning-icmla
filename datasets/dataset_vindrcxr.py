import torch
import pandas as pd
from PIL import Image
import os

class Dataset_VINDRCXR(torch.utils.data.Dataset):

  def __init__(self,df,root,transf,pathology):

    self.df=df
    self.pathology=pathology
    self.transf=transf

    self.root=root

  def __getitem__(self,idx):
    
    image_id=os.path.join(self.df['partition'].iloc[idx],self.df['image_id'].iloc[idx])

    
    imgs=Image.open(os.path.join(self.root,image_id+'.png'))
    labels=self.df[self.pathology].iloc[idx]
 
    imgs=self.transf(imgs)


    return imgs,labels
    
  def __len__(self):

    return len(self.df)