
import torch
import pandas as pd
from PIL import Image
import os





class Dataset_Chex(torch.utils.data.Dataset):

  def __init__(self,df,root,trans,pathology):

    self.df=df
    self.pathology=pathology
    self.root=root
    self.trans=trans
    
  def __getitem__(self,idx):
    

    
    png_path=self.df['Path'].iloc[idx]

    
    imgs=Image.open(os.path.join(self.root,png_path))

    labels=self.df[self.pathology].iloc[idx]
    
    if self.trans is not None:
        imgs=self.trans(imgs)
        
  
    return imgs,labels
    
  def __len__(self):

    return len(self.df)
  
