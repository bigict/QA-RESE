#!/usr/bin/env python3
# encoding: utf-8
import random,os, pickle, bz2,math,argparse,linecache
import numpy as np
from scipy import stats
import traceback
import torch,time
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataload import QARESEDataProvider
from model import QARESEModel

train_list = []
train_f = open("data_list/train_list.txt","r")
for line in train_f:
    train_list.append(line[:-1])
random.shuffle(train_list)

valid_f = open("data_list/valid_list.txt","r")
valid_list = []
for line in valid_f:
    valid_list.append(line[:-1])
random.shuffle(valid_list)

test_f = open("data_list/test_list.txt","r")
test_list = []
for line in test_f:
    test_list.append(line[:-1])
random.shuffle(test_list)

#save_para_path = "train_pth_save/"
save_para_path = "train_pth_save_new/"

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.0001 * (0.5 ** (epoch // 3))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def weights_init(m):
    for m in self.modules():
     if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
     elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class QARESE:
    def __init__(self):
        torch.cuda.set_device(1)
        self.device = torch.device('cuda:1')
        self.model = QARESEModel().to(self.device)
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.0001,betas=(0.9,0.999))

    def get_score(self,pred_arr,real_arr):
        G_PCC = np.corrcoef(pred_arr,real_arr)[0, 1]
        G_Spearman = stats.spearmanr(pred_arr,real_arr)[0]
        G_Kendall = stats.kendalltau(pred_arr,real_arr)[0]
        return G_PCC,G_Spearman,G_Kendall

    def train(self):
        self.running_loss=0.0
        self.model.train()
        for i in range(0,20):
           self.train_epoch(i)

    def train_epoch(self,epoch):
        start = time.time()
        adjust_learning_rate(self.optimizer,epoch)
        print('epoch:',epoch)
        self.model.train()
        cnt =0
        self.running_loss=0.0
        real_val,pred_val = [],[]
        for files in train_list:
                pro_name = files.split("/")[-2]
                Features=files+'/'+pro_name+".features.pkl"
                disFeatures=files+'/'+pro_name+".disfeatures.pkl"
                dires=files+'/'+pro_name+"_pkl/" 
                try:
                 if not os.path.exists(dires):continue 
                 
                 dpp = QARESEDataProvider(True,files,Features,disFeatures,dires)
                 if len(dpp) == 0:continue
                 for dict_features in dpp:
                  pred_global,real_global = self.model(dict_features['dim1'].float().to(self.device), dict_features['dim2'].float().to(self.device)),dict_features['gdt'].to(self.device)
                  real_val.append(real_global[0][0].item())
                  pred_val.append(pred_global[0][0].item())
                  loss = self.loss_fn(pred_global,real_global)
                  loss = loss*10
                  self.running_loss += float(loss.item())
                 cnt+=1
                 if cnt % 3500 == 0:
                      print("train epoch %s loss:%s"%(epoch,self.running_loss/cnt))
                 self.optimizer.zero_grad()
                 loss.backward()
                 self.optimizer.step()
                except Exception as e:
                 continue
        G_PCC, G_Spearman, G_Kendall = self.get_score(pred_val,real_val)
        print("'PCC': %s, 'Spearman': %s, 'Kendall': %s"%(G_PCC, G_Spearman, G_Kendall))
        real_val,pred_val = [],[]
        end = time.time()
        torch.save(self.model.state_dict(),save_para_path+"train_epoch%s.pth"%(epoch))
    
    
    def valid(self):
       self.model.eval()
       self.valid_epoch()

    def valid_epoch(self):
       with torch.no_grad():
         try:
           avg_G_PCC, avg_G_Spearman, avg_G_Kendall = [],[],[]
           for files in valid_list:
                tmp_real_val,tmp_pred_val = [],[]
                pro_name = files.split("/")[-2]
                Features=files+pro_name+".features.pkl"
                disFeatures=files+pro_name+".disfeatures.pkl"
                dires=files + pro_name + "_pkl/"

                dpp = QARESEDataProvider(False,files,Features,disFeatures,dires)

                for dict_features in dpp:
                  pred_global,real_global = self.model(dict_features['dim1'].float().to(self.device), dict_features['dim2'].float().to(self.device)),dict_features['gdt'].to(self.device)
                  loss = self.loss_fn(pred_global,real_global)
                  tmp_real_val.append(real_global[0][0].item())
                  tmp_pred_val.append(pred_global[0][0].item())
                G_PCC, G_Spearman, G_Kendall = self.get_score(tmp_pred_val,tmp_real_val)
                avg_G_PCC.append(G_PCC)
                avg_G_Spearman.append(G_Spearman)
                avg_G_Kendall.append(G_Kendall)
                print("'Protein Name':%s 'PCC': %s, 'Spearman': %s, 'Kendall': %s:"%(files,G_PCC, G_Spearman, G_Kendall))
           G_PCC, G_Spearman, G_Kendall = np.mean(avg_G_PCC),np.mean(avg_G_Spearman),np.mean(avg_G_Kendall)
           print("Valid Total: 'PCC': %s, 'Spearman': %s, 'Kendall': %s"%(G_PCC, G_Spearman, G_Kendall))
         except Exception as e:
             tmp = 'wrong'
    
    def test(self):
       self.model.eval()
       self.test_epoch()

    def test_epoch(self):
       with torch.no_grad():
         try:
           avg_G_PCC, avg_G_Spearman, avg_G_Kendall = [],[],[]
           for files in test_list:
                tmp_real_val,tmp_pred_val = [],[]
                cnt = 0
                pro_name = files.split("/")[-2]
                Features=files+pro_name+".features.pkl"
                disFeatures=files+pro_name+".disfeatures.pkl"
                dires = files+  pro_name +"_pkl/"

                dpp = QARESEDataProvider(False,files,Features,disFeatures,dires)
                for dict_features in dpp:
                  cnt += 1
                  pred_global,real_global = self.model(dict_features['dim1'].float().to(self.device), dict_features['dim2'].float().to(self.device)),dict_features['gdt'].to(self.device)
                  loss = self.loss_fn(pred_global,real_global)
                  tmp_real_val.append(real_global[0][0].item())
                  tmp_pred_val.append(pred_global[0][0].item())
                G_PCC, G_Spearman, G_Kendall = self.get_score(tmp_pred_val,tmp_real_val)
                avg_G_PCC.append(G_PCC)
                avg_G_Spearman.append(G_Spearman)
                avg_G_Kendall.append(G_Kendall)  
                print("'Protein Name':%s 'PCC': %s, 'Spearman': %s, 'Kendall': %s:"%(files,G_PCC, G_Spearman, G_Kendall))
           G_PCC, G_Spearman, G_Kendall = np.mean(avg_G_PCC),np.mean(avg_G_Spearman),np.mean(avg_G_Kendall)
           print("Test Total: 'PCC': %s, 'Spearman': %s, 'Kendall': %s"%(G_PCC, G_Spearman, G_Kendall))
         
         except Exception as e:
             tmp = 'wrong'

if __name__ == "__main__":
    qaRese = QARESE()
    qaRese.train()
    qaRese.valid()
    qaRese.test()
