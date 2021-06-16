import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch,pickle,os,random,linecache
from collections import Counter
import torch.nn.functional as F
import os, pickle, argparse
from torch.utils.data.dataloader import default_collate

def grep_gdt(val_path):
        strings = []
        strings = linecache.getlines(val_path)
        GDTTS='*'
        if(len(strings)>3):
           GDTTS   = strings[18]
           GDTTS   = GDTTS[14:20]
           return GDTTS
        else:
           return "error"


def calc_gdt(struct, npz_name):
    native_pdb = pdb_path + npz_name +'.pdb'
    tm_file = "/home/dangmingai/QARESE_code/tmp.txt"
    os.system("/mnt/data3/softwares/hhsuite-2.0.16-linux-x86_64/scripts/hhpred/bin/TMscore %s %s > %s"%(struct,native_pdb,tm_file))
    gdt_val = grep_gdt(tm_file)
    return gdt_val

Residue_arr = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y','X']

def grep_sequence_oh(sequence):
    sequence_oh = np.zeros((len(sequence), len(Residue_arr)))
    Seq_arr = sequence.upper()
    for No, residue in enumerate(Seq_arr):
        if residue not in Residue_arr: 
            residue = "X"
        sequence_oh[No, Residue_arr.index(residue)] = 1
    return sequence_oh

def grep_sequence_rp(sequence_size):
    rp = np.linspace(0, 1, num=sequence_size).reshape(sequence_size, -1)
    return rp


class QARESEDataset(Dataset):
    def __init__(self, Features, disFeatures, dires,train_flag):
        self.Features = Features
        self.disFeatures = disFeatures
        self.dires = dires
        self.sequence_gene_fea = {'oh': 'dim1', 'rp': 'dim1','pssm': 'dim1','ss3': 'dim1', 'acc': 'dim1','ccmpredz': 'dim2', 'alnstats_matrix': 'dim2', 'dis_matrix': 'dim2'}
        self.structures_gene_fea = {'gdt':'dim1','ss3': 'dim1','rsa': 'dim1', 'cbcb': 'dim2','caca': 'dim2','no': 'dim2'}

        self.structs = []
        for files in os.listdir(dires):
            if train_flag:
                if os.path.isfile(dires+files) and files.split('.')[1][0]>'3':
                    self.structs.append(dires+files)
            else:
                if os.path.isfile(dires+files):
                    self.structs.append(dires+files)

    def __len__(self):
        return len(self.structs)

    def grep_coll_features(self):
        Features_datas = pickle.load(open(self.Features, 'rb'))
        disFeatures_datas = pickle.load(open(self.disFeatures, 'rb'))
        coll_features = {'sequence': Features_datas['sequence'],'oh': grep_sequence_oh(Features_datas['sequence']),'rp': grep_sequence_rp(len(Features_datas['sequence'])),'acc': Features_datas['acc'],'pssm': Features_datas['pssm'],'ss3': Features_datas['ss3'],'ccmpredz': Features_datas['ccmpredz'],'alnstats_matrix': Features_datas['alnstats_matrix'],'dis_matrix': disFeatures_datas['dis_matrix']}
        return coll_features

    def __getitem__(self, cnt):
        if torch.is_tensor(cnt): cnt = cnt.tolist()
        structs_file = self.structs[cnt]
        dict_features = {"dim1": None, "dim2": None}
        coll_features = self.grep_coll_features()
        for key in self.sequence_gene_fea:
            size = self.sequence_gene_fea[key]
            vals = coll_features[key]
            if size=='dim2': vals = vals.reshape((vals.shape[0], vals.shape[1], -1))
            if dict_features[size] is None:
                dict_features[size] = vals
            else:
                dict_features[size] = np.concatenate((dict_features[size], vals), axis=-1)
        struct_features = pickle.load(open(structs_file, 'rb'))
        val = np.zeros((1))
        val[0]=structs_file.split('/')[-1].split('_')[0]
        struct_features['gdt'] = val.astype(np.float16)
        for key in self.structures_gene_fea:
            if key == 'gdt':continue
            size = self.structures_gene_fea[key]
            vals = struct_features[key]
            dict_features[size] = np.concatenate((dict_features[size], vals), axis=-1)

        dict_features = np.nan_to_num(dict_features)
        dict_features['dim1'] = dict_features['dim1'].transpose((1,0)).astype(np.float)
        dict_features['dim2'] = dict_features['dim2'].transpose((2,0,1)).astype(np.float)
        dict_features['gdt'] = struct_features['gdt']
        structs_coll = {
            'struct': structs_file,
        }

        #return dict_features,structs_coll
        return dict_features


class QARESEDataProvider(DataLoader):
    def __init__(self,train,name,Features,disFeatures,dires):
        if train:
            self.dataset = QARESEDataset(Features,disFeatures,dires,True)
            super(QARESEDataProvider, self).__init__(
                self.dataset, batch_size=1, shuffle=True,num_workers=1,pin_memory=False)
        else:
            self.dataset = QARESEDataset(Features,disFeatures,dires,False)
            super(QARESEDataProvider, self).__init__(
                self.dataset, batch_size=1, shuffle=False, num_workers=4,pin_memory=True)

