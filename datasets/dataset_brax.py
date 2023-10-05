import torch
import pandas as pd
from PIL import Image
import os
import numpy as np

"""# Dataset  - Binary Classification"""

class Dataset_BRAX(torch.utils.data.Dataset):

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
  


"""BRAX - XRAY FEATURES """

  
class Dataset_BRAX_MultiHead_xray_interpol(torch.utils.data.Dataset):

  def __init__(self,df,root_orig,root_interpol,transf,pathology):

    self.df=df
    self.pathology=pathology
    self.transf=transf
    self.root_orig=root_orig
    self.root_interpol=root_interpol


  def __getitem__(self,idx):
    
    png_path=self.df['PngPath'].iloc[idx]
    
    img_orig=Image.open(os.path.join(self.root_orig,png_path))
    img_interpol=Image.open(os.path.join(self.root_interpol,png_path))
    
    label_dis=self.df[self.pathology].iloc[idx]
    labels_xray=self.df['xray_labels'].iloc[idx]
    
    if self.transf is not None:
      img_orig=self.transf(img_orig)
      img_interpol=self.transf(img_interpol)
    
    return img_orig,img_interpol,label_dis,labels_xray
    
  def __len__(self):

    return len(self.df)
  
  
class Dataset_BRAX_ablation(torch.utils.data.Dataset):

  def __init__(self,df,root_orig,root_interpol,transf,pathology):

    self.df=df
    self.pathology=pathology
    self.transf=transf
    self.root_orig=root_orig
    self.root_interpol=root_interpol


  def __getitem__(self,idx):
    
    png_path=self.df['PngPath'].iloc[idx]

    img_interpol=Image.open(os.path.join(self.root_interpol,png_path))
    img_orig=Image.open(os.path.join(self.root_orig,png_path))
    
    
    labels=self.df[self.pathology].iloc[idx]
    
    img_orig=self.transf(img_orig)
    img_interpol=self.transf(img_interpol)

    return img_orig,img_interpol,labels
    
  def __len__(self):

    return len(self.df)