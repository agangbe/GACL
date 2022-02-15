# --------- GACL meathod ------------
# Time 2022 

import sys,os
from numpy.matrixlib.defmatrix import matrix
sys.path.append(os.getcwd())
from Process.process import * 
#from Process.process_user import *
import torch as th
import torch.nn as nn
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from others.earlystopping import EarlyStopping
from torch_geometric.data import DataLoader
from tqdm import tqdm
from Process.rand5fold import *
from others.evaluate import *
from torch_geometric.nn import GCNConv
import copy
import random


def setup_seed(seed):
     th.manual_seed(seed)
     th.cuda.manual_seed_all(seed) 
     np.random.seed(seed)
     random.seed(seed)
     th.backends.cudnn.deterministic = True

setup_seed(2022)


class hard_fc(th.nn.Module):
    def __init__(self, d_in,d_hid, DroPout=0):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(DroPout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x

class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=0.3, emb_name='hard_fc1.'): # T15: epsilon = 0.2
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = th.norm(param.grad)
                if norm != 0 and not th.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='hard_fc1.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class GCN_Net(th.nn.Module): 
    def __init__(self,in_feats,hid_feats,out_feats): 
        super(GCN_Net, self).__init__() 
        self.conv1 = GCNConv(in_feats, hid_feats)
        self.conv2 = GCNConv(hid_feats, out_feats)
        self.fc=th.nn.Linear(2*out_feats,4)
        self.hard_fc1 = hard_fc(out_feats, out_feats)
        self.hard_fc2 = hard_fc(out_feats, out_feats) # optional

    def forward(self, data):
        init_x0, init_x, edge_index1, edge_index2 = data.x0, data.x, data.edge_index, data.edge_index2
        
        x1 = self.conv1(init_x0, edge_index1)
        x1 = F.relu(x1)
        x1 = self.conv2(x1, edge_index1)
        x1 = F.relu(x1)
        x1 = scatter_mean(x1, data.batch, dim=0)
        x1_g = x1
        x1 = self.hard_fc1(x1)
        x1_t = x1
        x1 = th.cat((x1_g, x1_t), 1)

        x2 = self.conv1(init_x, edge_index2)
        x2 = F.relu(x2)
        x2 = self.conv2(x2, edge_index2)
        x2 = F.relu(x2)
        x2 = scatter_mean(x2, data.batch, dim=0)
        x2_g = x2
        x2 = self.hard_fc1(x2)
        x2_t = x2 
        x2 = th.cat((x2_g, x2_t), 1)
        x = th.cat((x1, x2), 0)
        y = th.cat((data.y1, data.y2), 0)

        x_T = x.t()
        dot_matrix = th.mm(x, x_T)
        x_norm = th.norm(x, p=2, dim=1)
        x_norm = x_norm.unsqueeze(1)
        norm_matrix = th.mm(x_norm, x_norm.t())
        
        t = 0.3 # pheme: t = 0.6
        cos_matrix = (dot_matrix / norm_matrix) / t
        cos_matrix = th.exp(cos_matrix)
        diag = th.diag(cos_matrix)
        cos_matrix_diag = th.diag_embed(diag)
        cos_matrix = cos_matrix - cos_matrix_diag
        y_matrix_T = y.expand(len(y), len(y))
        y_matrix = y_matrix_T.t()
        y_matrix = th.ne(y_matrix, y_matrix_T).float()
        #y_matrix_list = y_matrix.chunk(3, dim=0)
        #y_matrix = y_matrix_list[0]
        neg_matrix = cos_matrix * y_matrix
        neg_matrix_list = neg_matrix.chunk(2, dim=0)
        #neg_matrix = neg_matrix_list[0]
        pos_y_matrix = y_matrix * (-1) + 1
        pos_matrix_list = (cos_matrix * pos_y_matrix).chunk(2,dim=0)
        #print('cos_matrix: ', cos_matrix.shape, cos_matrix)
        #print('pos_y_matrix: ', pos_y_matrix.shape, pos_y_matrix)
        pos_matrix = pos_matrix_list[0]
        #print('pos shape: ', pos_matrix.shape, pos_matrix)
        neg_matrix = (th.sum(neg_matrix, dim=1)).unsqueeze(1)
        sum_neg_matrix_list = neg_matrix.chunk(2, dim=0)
        p1_neg_matrix = sum_neg_matrix_list[0]
        p2_neg_matrix = sum_neg_matrix_list[1]
        neg_matrix = p1_neg_matrix
        #print('neg shape: ', neg_matrix.shape)
        div = pos_matrix / neg_matrix 
        div = (th.sum(div, dim=1)).unsqueeze(1)  
        div = div / batchsize
        log = th.log(div)
        SUM = th.sum(log)
        cl_loss = -SUM

        x = self.fc(x) 
        x = F.log_softmax(x, dim=1)

        return x, cl_loss, y


def train_GCN(x_test, x_train,lr, weight_decay,patience,n_epochs,batchsize,dataname):
    model = GCN_Net(768,64,64).to(device) 
    fgm = FGM(model)
    for para in model.hard_fc1.parameters():
        para.requires_grad = False
    for para in model.hard_fc2.parameters():
        para.requires_grad = False
    optimizer = th.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    
    # optional ------ S1 ----------
    for para in model.hard_fc1.parameters():
        para.requires_grad = True
    for para in model.hard_fc2.parameters():
        para.requires_grad = True
    #optimizer_hard = th.optim.Adam(model.hard_fc.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_hard = th.optim.SGD([{'params': model.hard_fc1.parameters()},
                                    {'params': model.hard_fc2.parameters()}], lr=0.001)

    model.train() 
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=patience, verbose=True) 

    for epoch in range(n_epochs): 
        traindata_list, testdata_list = loadData(dataname, x_train, x_test, droprate=0.4) # T15 droprate = 0.1
        train_loader = DataLoader(traindata_list, batch_size=batchsize, shuffle=True, num_workers=5)
        test_loader = DataLoader(testdata_list, batch_size=batchsize, shuffle=True, num_workers=5)     
        avg_loss = [] 
        avg_acc = []
        batch_idx = 0
        tqdm_train_loader = tqdm(train_loader)
        NUM=1
        beta=0.001
        for Batch_data in tqdm_train_loader:
            Batch_data.to(device)
            out_labels, cl_loss, y = model(Batch_data) 
            finalloss = F.nll_loss(out_labels,y)
            loss = finalloss + beta*cl_loss
            avg_loss.append(loss.item())
            ##------------- S1 ---------------##
            '''
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            avg_loss.append(loss.item())
            optimizer.step()
            epsilon = 3
            loss_ad = epsilon/(finalloss + 0.001*cl_loss)
            print('loss_ad: ', loss_ad)
            optimizer_hard.zero_grad()
            loss_ad.backward()
            optimizer_hard.step()
            '''
            ##--------------------------------##

            ##------------- S2 ---------------##
            optimizer.zero_grad()
            loss.backward()
            fgm.attack()
            out_labels, cl_loss, y = model(Batch_data) 
            finalloss = F.nll_loss(out_labels,y)
            loss_adv = finalloss + beta*cl_loss
            loss_adv.backward()
            fgm.restore()
            optimizer.step()
            ##--------------------------------##


            _, pred = out_labels.max(dim=-1)
            correct = pred.eq(y).sum().item()
            train_acc = correct / len(y) 
            avg_acc.append(train_acc)
            print("Epoch {:05d} | Batch{:02d} | Train_Loss {:.4f}| Train_Accuracy {:.4f}".format(epoch, batch_idx,loss.item(),train_acc))
            batch_idx = batch_idx + 1
            NUM += 1
            #print('train_loss: ', loss.item())
        train_losses.append(np.mean(avg_loss))
        train_accs.append(np.mean(avg_acc))
        
        temp_val_losses = []
        temp_val_accs = []
        temp_val_Acc_all, temp_val_Acc1, temp_val_Prec1, temp_val_Recll1, temp_val_F1, \
        temp_val_Acc2, temp_val_Prec2, temp_val_Recll2, temp_val_F2, \
        temp_val_Acc3, temp_val_Prec3, temp_val_Recll3, temp_val_F3, \
        temp_val_Acc4, temp_val_Prec4, temp_val_Recll4, temp_val_F4 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
        model.eval() 
        tqdm_test_loader = tqdm(test_loader)
        for Batch_data in tqdm_test_loader:
            Batch_data.to(device)
            val_out, val_cl_loss, y = model(Batch_data)
            val_loss = F.nll_loss(val_out, y)
            temp_val_losses.append(val_loss.item())
            _, val_pred = val_out.max(dim=1)
            correct = val_pred.eq(y).sum().item()
            val_acc = correct / len(y) 
            Acc_all, Acc1, Prec1, Recll1, F1, Acc2, Prec2, Recll2, F2, Acc3, Prec3, Recll3, F3, Acc4, Prec4, Recll4, F4 = evaluation4class(
                val_pred, y) 
            temp_val_Acc_all.append(Acc_all), temp_val_Acc1.append(Acc1), temp_val_Prec1.append(
                Prec1), temp_val_Recll1.append(Recll1), temp_val_F1.append(F1), \
            temp_val_Acc2.append(Acc2), temp_val_Prec2.append(Prec2), temp_val_Recll2.append(
                Recll2), temp_val_F2.append(F2), \
            temp_val_Acc3.append(Acc3), temp_val_Prec3.append(Prec3), temp_val_Recll3.append(
                Recll3), temp_val_F3.append(F3), \
            temp_val_Acc4.append(Acc4), temp_val_Prec4.append(Prec4), temp_val_Recll4.append(
                Recll4), temp_val_F4.append(F4)
            temp_val_accs.append(val_acc)
        val_losses.append(np.mean(temp_val_losses))
        val_accs.append(np.mean(temp_val_accs))
        print("Epoch {:05d} | Val_Loss {:.4f} | Val_Accuracy {:.4f}".format(epoch, np.mean(avg_loss), np.mean(temp_val_losses),
                                                                           np.mean(temp_val_accs)))
        res = ['acc:{:.4f}'.format(np.mean(temp_val_Acc_all)),
               'C1:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc1), np.mean(temp_val_Prec1),
                                                       np.mean(temp_val_Recll1), np.mean(temp_val_F1)),
               'C2:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc2), np.mean(temp_val_Prec2),
                                                       np.mean(temp_val_Recll2), np.mean(temp_val_F2)),
               'C3:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc3), np.mean(temp_val_Prec3),
                                                       np.mean(temp_val_Recll3), np.mean(temp_val_F3)),
               'C4:{:.4f},{:.4f},{:.4f},{:.4f}'.format(np.mean(temp_val_Acc4), np.mean(temp_val_Prec4),
                                                       np.mean(temp_val_Recll4), np.mean(temp_val_F4))]
        print('results:', res)
      
        if epoch > 25:
            early_stopping(np.mean(temp_val_losses), np.mean(temp_val_accs), np.mean(temp_val_F1), np.mean(temp_val_F2),
                        np.mean(temp_val_F3), np.mean(temp_val_F4), model, 'GACL', dataname)
        accs =np.mean(temp_val_accs)
        F1 = np.mean(temp_val_F1)
        F2 = np.mean(temp_val_F2)
        F3 = np.mean(temp_val_F3)
        F4 = np.mean(temp_val_F4)
        if early_stopping.early_stop:
            print("Early stopping")
            accs=early_stopping.accs
            F1=early_stopping.F1
            F2 = early_stopping.F2
            F3 = early_stopping.F3
            F4 = early_stopping.F4
            break
    return accs,F1,F2,F3,F4


##---------------------------------main---------------------------------------
scale = 1
lr=0.0005 * scale
weight_decay=1e-4
patience=10
n_epochs=200
batchsize=120  
datasetname='Twitter16' # (1)Twitter15  (2)pheme  (3)weibo
#model="GCN" 
device = th.device('cuda:4' if th.cuda.is_available() else 'cpu')
test_accs = [] 
NR_F1 = [] # NR
FR_F1 = [] # FR
TR_F1 = [] # TR
UR_F1 = [] # UR

data_path = './data/twitter16/'
laebl_path = './data/Twitter16_label_All.txt'

fold0_x_test, fold0_x_train, \
fold1_x_test,  fold1_x_train,\
fold2_x_test, fold2_x_train, \
fold3_x_test, fold3_x_train, \
fold4_x_test,fold4_x_train = load5foldData(datasetname,data_path,laebl_path)

print('fold0 shape: ', len(fold0_x_test), len(fold0_x_train))
print('fold1 shape: ', len(fold1_x_test), len(fold1_x_train))
print('fold2 shape: ', len(fold2_x_test), len(fold2_x_train))
print('fold3 shape: ', len(fold3_x_test), len(fold3_x_train))
print('fold4 shape: ', len(fold4_x_test), len(fold4_x_train))


accs0, F1_0, F2_0, F3_0, F4_0 = train_GCN(fold0_x_test,fold0_x_train,lr,weight_decay, patience,n_epochs,batchsize,datasetname)
accs1, F1_1, F2_1, F3_1, F4_1 = train_GCN(fold1_x_test,fold1_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
accs2, F1_2, F2_2, F3_2, F4_2 = train_GCN(fold2_x_test,fold2_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
accs3, F1_3, F2_3, F3_3, F4_3 = train_GCN(fold3_x_test,fold3_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
accs4, F1_4, F2_4, F3_4, F4_4 = train_GCN(fold4_x_test,fold4_x_train,lr,weight_decay,patience,n_epochs,batchsize,datasetname)
test_accs.append((accs0+accs1+accs2+accs3+accs4)/5) 
NR_F1.append((F1_0+F1_1+F1_2+F1_3+F1_4)/5) 
FR_F1.append((F2_0 + F2_1 + F2_2 + F2_3 + F2_4) / 5) 
TR_F1.append((F3_0 + F3_1 + F3_2 + F3_3 + F3_4) / 5) 
UR_F1.append((F4_0 + F4_1 + F4_2 + F4_3 + F4_4) / 5)
print("AVG_result: {:.4f}|UR F1: {:.4f}|NR F1: {:.4f}|TR F1: {:.4f}|FR F1: {:.4f}".format(sum(test_accs), sum(NR_F1), sum(FR_F1), sum(TR_F1), sum(UR_F1)))
