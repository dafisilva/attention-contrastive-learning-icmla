"""# Model"""

import torch.nn as nn
import torch
import torchvision

import sys
sys.path.append('script')
from overloaded_models.resnet import *

device='cuda:0' if torch.cuda.is_available() else 'cpu'
softmax=nn.Softmax(dim=1)
mse=nn.MSELoss(reduction='sum')
cross=nn.CrossEntropyLoss()
sigmoid=nn.Sigmoid()

        
class Tiny_dis4(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        downsample1 = nn.Sequential(
                nn.Conv2d(64, 128,kernel_size=1,stride=2),
                 nn.BatchNorm2d(128),
            )

        self.enc1=BasicBlock(64,128,2,downsample=downsample1)
        self.enc2=BasicBlock(128,128,1)
        downsample2 = nn.Sequential(
                nn.Conv2d(128, 256,kernel_size=1,stride=2),
                 nn.BatchNorm2d(256),
            )
        
        self.enc3=BasicBlock(128,256,2,downsample=downsample2)
        self.enc4=BasicBlock(256,256,1)
        self.avgpool=nn.AdaptiveMaxPool2d(1)
        self.linear=nn.Sequential(nn.Flatten(),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,1,bias=True))
        
    def forward(self,x):
        
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        
        feat=self.enc1(x)
        feat=self.enc2(feat)
        feat=self.enc3(feat)
        feat=self.enc4(feat)
        
        feat=self.avgpool(feat)
        
        logits=self.linear(feat)
        
        return logits
    
class Tiny_MH(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        downsample1 = nn.Sequential(
                nn.Conv2d(64, 128,kernel_size=1,stride=2),
                 nn.BatchNorm2d(128),
            )
        downsample1_dis = nn.Sequential(
                nn.Conv2d(64, 128,kernel_size=1,stride=1),
                 nn.BatchNorm2d(128),
            )
        downsample1_x = nn.Sequential(
                nn.Conv2d(64, 128,kernel_size=1,stride=1),
                 nn.BatchNorm2d(128),
            )
        self.enc1=BasicBlock(64,128,2,downsample=downsample1)
        self.enc2_x=BasicBlock(64,128,1,downsample=downsample1_x)
        
        self.enc2_dis=BasicBlock(64,128,1,downsample=downsample1_dis)
        
        downsample2 = nn.Sequential(
                nn.Conv2d(128, 256,kernel_size=1,stride=2),
                 nn.BatchNorm2d(256),
            )
        
        self.enc3=BasicBlock(128,256,2,downsample=downsample2)
        self.enc4=BasicBlock(256,256,1)
        self.avgpool=nn.AdaptiveMaxPool2d(1)
        self.fc_x=nn.Sequential(nn.Flatten(),nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128,256,bias=True))
        self.linear_x=nn.Sequential(nn.ReLU(),
                                    nn.Linear(256,3,bias=True))
        self.linear_dis=nn.Sequential(nn.Flatten(),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,1,bias=True))
        
        
  
    def forward(self,x):
        
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        
        feat=self.enc1(x)
        feat_dis=self.enc2_dis(feat[:,:64,:,:])
        feat_x=self.enc2_x(feat[:,64:,:,:])
        
        
        
        feat_dis=self.enc3(feat_dis)
        feat_dis=self.enc4(feat_dis)
        
        
        feat_dis=self.avgpool(feat_dis)
        feat_x=self.fc_x(self.avgpool(feat_x))
        
        logits_dis=self.linear_dis(feat_dis)
        logits_x=self.linear_x(feat_x)
        return logits_dis,logits_x
    
    

class Attention_Module(nn.Module):
    def __init__(self,
                 n_channels:int=64):
        super().__init__()

        small_channel=int(n_channels/8)
        self.query=nn.Conv2d(n_channels,small_channel,kernel_size=1,stride=1)
        self.key=nn.Conv2d(n_channels,small_channel,kernel_size=1,stride=1)
        self.value=nn.Conv2d(n_channels,small_channel,kernel_size=1,stride=1)
        self.softmax=nn.Softmax(2)
        self.coef=nn.Parameter(torch.Tensor([0]))
        self.attn=nn.Conv2d(small_channel,n_channels,kernel_size=1,stride=1)
        
        
    def forward(self,x):
        
        batch,C,H,W=x.size()
        N=H*W
        query=self.query(x).view(batch,-1,N)
        key=self.key(x).view(batch,-1,N).permute(0,2,1)
        value=self.value(x).view(batch,-1,N)
        
        
        
        att_map=torch.bmm(key,query)
        
        att_map=self.softmax(att_map)
        
        
        
        value=torch.bmm(value,att_map).view(batch,-1,H,W)
        
        self_attn=self.attn(value)
        
        out=self.coef*self_attn+x
        return out,att_map
        
    
    

class Tiny_MH_EP_AM_interpol(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        downsample1 = nn.Sequential(
                nn.Conv2d(64, 128,kernel_size=1,stride=2),
                 nn.BatchNorm2d(128),
            )
        downsample1_sep = nn.Sequential(
                nn.Conv2d(64, 128,kernel_size=1,stride=1),
                 nn.BatchNorm2d(128),
            )
        self.enc1=BasicBlock(64,128,2,downsample=downsample1)
        
        self.attention_dis=Attention_Module(64)
        self.attention_x=Attention_Module(64)
        
        self.enc2_x=BasicBlock(64,128,1,downsample=downsample1_sep)
        
        self.enc2_dis=BasicBlock(64,128,1,downsample=downsample1_sep)
        
        downsample2 = nn.Sequential(
                nn.Conv2d(128, 256,kernel_size=1,stride=2),
                 nn.BatchNorm2d(256),
            )
        
        self.enc3=BasicBlock(128,256,2,downsample=downsample2)
        self.enc4=BasicBlock(256,256,1)
        self.avgpool=nn.AdaptiveMaxPool2d(1)
        self.fc_x=nn.Sequential(nn.Flatten(),nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(128,256,bias=True))
        self.linear_x=nn.Sequential(nn.ReLU(),
                                    nn.Linear(256,4,bias=True))
        self.linear_dis=nn.Sequential(nn.Flatten(),nn.ReLU(),nn.Dropout(0.2),nn.Linear(256,1,bias=True))
        
        self.flatten=nn.Flatten()
        
    def mse_eq_prob(self,out_m,b,task):
        
        if task=='xray':
            target=torch.Tensor([[1/4]*4]*b).to(device)
            target=target+torch.randn_like(target)*0.01
            out_m=softmax(out_m)
        else:
            target=torch.Tensor([[1/2]*1]*b).to(device)
            target=target+torch.randn_like(target)*0.01
            out_m=sigmoid(out_m)
            
        return mse(out_m,target)    
    def forward(self,x,task='classification'):
        
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpool(x)
        
        feat=self.enc1(x)
        
        
        attn_dis,map_dis=self.attention_dis(feat[:,:64,:,:])
        attn_x,map_x=self.attention_x(feat[:,64:,:,:])
        
        feat_dis=self.enc2_dis(attn_dis)
        feat_x=self.enc2_x(attn_x)
        
        feat_dis=self.enc3(feat_dis)
        feat_dis=self.enc4(feat_dis)
        
        
        feat_dis=self.avgpool(feat_dis)
        feat_x=self.fc_x(self.avgpool(feat_x))
        
        logits_dis=self.linear_dis(feat_dis)
        logits_x=self.linear_x(feat_x)
        
        if task=='classification':
            
            return logits_dis,logits_x,torch.flatten(map_dis,start_dim=1),torch.flatten(map_x,start_dim=1)
        
        elif task=="inference":
            
            return attn_dis,attn_x
        else:
            fake_logits_dis=self.linear_dis(feat_x)
            fake_logits_x=self.linear_x(self.flatten(feat_dis))
            
            eq_prob_loss_dis=self.mse_eq_prob(fake_logits_dis,fake_logits_dis.shape[0],'dis')
            eq_prob_loss_x=self.mse_eq_prob(fake_logits_x,fake_logits_x.shape[0],'xray')
            
            eq_prob_loss=eq_prob_loss_dis+eq_prob_loss_x
            
            
            return eq_prob_loss
