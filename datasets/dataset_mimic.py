import numpy as np
import torch
from PIL import Image
import pandas as pd
import os



class Dataset_MIMIC(torch.utils.data.Dataset):

  def __init__(self,df,root,transf,pathology):

    self.df=df
    self.pathology=pathology
    self.transf=transf
    self.root=root


  def __getitem__(self,idx):
    
    png_path=self.df['PngPath'].iloc[idx]

    imgs=Image.open(os.path.join(self.root,png_path))

    labels=self.df[self.pathology].iloc[idx]
    
    imgs=self.transf(imgs)

    return imgs,labels
    
  def __len__(self):

    return len(self.df)