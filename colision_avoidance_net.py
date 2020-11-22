#!/usr/bin/env python
# coding: utf-8

# In[10]:


import torch,itertools,argparse,os,time,sys,random
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from typing import Type, Any, Callable, Union, List, Optional
from torchsummary import summary
import seaborn as sns



# In[ ]:


parser = argparse.ArgumentParser(description='Train Implementation')
parser.add_argument('--num_layers', nargs='+', type=int, default=[2, 2, 2], help='num layers')
parser.add_argument('--num_nodes', nargs='+', type=int, default=[40, 40, 40], help='num nodes')
parser.add_argument('--index', type=int, default=0, help='index')
args = parser.parse_args()


# In[11]:


mean=np.load('mean.npy').tolist()
std=np.load('std.npy').tolist()
mean_test=np.load('mean_test.npy').tolist()
std_test=np.load('std_test.npy').tolist()


# In[12]:


class CustomDataset(Dataset):
    def __init__(self,path):
        xy = np.loadtxt(path,
                        delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.tensor(xy[:, 0:5])
        xy=xy.astype('int_')
        self.y_data = torch.tensor(xy[:, 5])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# In[13]:


batch_size=300
lr=0.001

num_layers=args.num_layers
_nodes=args.num_nodes

model_char="{}_{}_{}_{}_{}_{}".format(_nodes[0],_nodes[1],_nodes[2],num_layers[0],num_layers[1],num_layers[2])
log_file = open('./res_log/'+model_char+'.txt','w')
sys.stdout = log_file

train_dataset = CustomDataset('norm_data_train_uniform_ext.csv')
train_loader = DataLoader(dataset=train_dataset,pin_memory=True,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=60,drop_last=True)
test_dataset = CustomDataset('norm_data_test_uniform_ext.csv')
test_loader = DataLoader(dataset=test_dataset,pin_memory=True,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=60,drop_last=True)


# In[14]:


class FClayer(nn.Module):
    def __init__(self, innodes: int, nodes: int):
        super(FClayer, self).__init__()
        self.fc=nn.Linear(innodes,nodes)
        self.act=nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x: Tensor) -> Tensor:
        out=self.fc(x)
        out=self.act(out)
        return out


# In[15]:


class WaveNET(nn.Module):
    def __init__(self, block: Type[Union[FClayer]], planes: List[int], nodes: List[int], num_classes: int = 3
                ) -> None:
        super(WaveNET, self).__init__()
        self.innodes=5
        
        self.layer1=self._make_layer(block, planes[0], nodes[0])
        self.layer2=self._make_layer(block, planes[1], nodes[1])
        self.layer3=self._make_layer(block, planes[2], nodes[2])
        
        self.fin_fc=nn.Linear(self.innodes,num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
    
    def _make_layer(self, block: Type[Union[FClayer]], planes: int, nodes: int) -> nn.Sequential:

        layers = []
        layers.append(block(self.innodes, nodes))
        self.innodes = nodes
        for _ in range(1, planes):
            layers.append(block(self.innodes, nodes))

        return nn.Sequential(*layers)

        
    def _forward_impl(self, x: Tensor) -> Tensor:
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fin_fc(x)
        
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


# In[16]:


def Model(block, planes, **kwargs):
    model = WaveNET(block, planes, **kwargs)
    return model


# In[17]:


model=WaveNET(FClayer,num_layers,_nodes).cuda()


# In[18]:
summary(model,(1,5))

criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20)


# In[ ]:


saving_path="./res_model/"
trn_loss_list = []
val_loss_list = []
val_acc_list = []
total_epoch=50

model_name=""
patience=10
start_early_stop_check=0
saving_start_epoch=10

for epoch in range(total_epoch):
    trn_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs=inputs.cuda()
            labels=labels.cuda()
        # grad init
        optimizer.zero_grad()
        # forward propagation
        output= model(inputs)
        # calculate loss
        loss=criterion(output, labels)
        # back propagation 
        loss.backward()
        # weight update
        optimizer.step()
        
        # trn_loss summary
        trn_loss += loss.item()
        # del (memory issue)
        del loss
        del output
    with torch.no_grad():
        val_loss = 0.0
        cor_match = 0
        for j, val in enumerate(test_loader):
            val_x, val_label = val
            if torch.cuda.is_available():
                val_x = val_x.cuda()
                val_label =val_label.cuda()
            val_output = model(val_x)
            v_loss = criterion(val_output, val_label)
            val_loss += v_loss
            _, predicted=torch.max(val_output,1)
            cor_match+=np.count_nonzero(predicted.cpu().detach()==val_label.cpu().detach())
    del val_output
    del v_loss
    del predicted
    
    scheduler.step()
    
    
    
    trn_loss_list.append(trn_loss/len(train_loader))
    val_loss_list.append(val_loss/len(test_loader))
    val_acc=cor_match/(len(test_loader)*batch_size)
    val_acc_list.append(val_acc)
    now = time.localtime()
    print ("%04d/%02d/%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec))

    print("epoch: {}/{} | trn loss: {:.4f} | val loss: {:.4f} | val accuracy: {:.4f}% \n".format(
                epoch+1, total_epoch, trn_loss / len(train_loader), val_loss / len(test_loader), val_acc*100
            ))
    
    
    if epoch+1>2:
        if val_loss_list[-1]>val_loss_list[-2]:
            start_early_stop_check=1
    else:
        val_loss_min=val_loss_list[-1]
        
    if start_early_stop_check:
        early_stop_temp=val_loss_list[-patience:]
        if all(early_stop_temp[i]<early_stop_temp[i+1] for i in range (len(early_stop_temp)-1)):
            print("Early stop!")
            break
            
#     if epoch+1>saving_start_epoch:
#         if val_loss_list[-1]<val_loss_min:
#             if os.path.isfile(model_name):
#                 os.remove(model_name)
#             val_loss_min=val_loss_list[-1]
#             model_name=saving_path+"Custom_model_"+model_char+"_{:.3f}".format(val_loss_min)
#             torch.save(model, model_name)
#             print("Model replaced and saved as ",model_name)


# In[ ]:


model_name=saving_path+"Custom_model_"+model_char+"_fin"
torch.save(model, model_name)


# In[ ]:


saving_img_path="./res_img/"


# In[ ]:


# initial setting

import numpy as np
import matplotlib.pyplot as plt

Deg2Rad = np.pi/180
Rad2Deg = 1/Deg2Rad

dt = 0.1              # control frequency
tf = 15               # final time
g = 9.8
K_alt = .8*2          # hdot loop gain    
RoC = 20              # maximum rate of climb (max. of hdot)
AoA0 = -1.71*Deg2Rad     # zero lift angle of attack
Acc2AoA = 0.308333*Deg2Rad  # 1m/s^2 ACC corresponds to 0.308333deg AOA 
zeta_ap = 0.7         # pitch acceleration loop damping
omega_ap = 4          # pitch acceleration loop bandwidth

dist_sep = 100        # near mid-air collision range

t = np.arange(0, tf, dt)
N = len(t)
# hdot loop dynamics definition

def int_model(z, t, hdot_cmd):                          # computes state derivatives  
    a, adot, h, hdot, R = z                           # state vector: a (pitch acc), adot, h (alt), hdot, R (ground-track range)
    gamma=np.arcsin(hdot/Vm)                          # fight path angle
    ac = K_alt * (hdot_cmd - hdot) + g/np.cos(gamma)  # pitch acceleration command
    ac = np.clip(ac, -30, 30)                         # maneuver limit
  
    addot = omega_ap*omega_ap*(ac-a) - 2*zeta_ap*omega_ap*adot
    hddot = a*np.cos(gamma) - g
    Rdot = Vm*np.cos(gamma)
    return np.array([adot, addot, hdot, hddot, Rdot]) # returns state derivatives 


# In[ ]:


# player initial conditions
total_sim=500

hdot_flag=0
res_Y = np.zeros(((N,7,total_sim)))                       # print-out data


hdot_res_cmd=[]
while True:
    hdot_res=[]
    hdot_res.append(0)
    time_temp=0
    errcmd=0
    insight=0
    
    
    hm0 = 1000                                                     # initial altitude
    Vm = 200                                                       # initial speed
    gamma0 = 0*Deg2Rad                                             # initial flight path angle
    Pm_NED = np.array([0, 0, -hm0])                                # initial NED position
    Vm_NED = np.array([Vm*np.cos(gamma0), 0, -Vm*np.sin(gamma0)])  # initial NED velocity

    # state variable: [a, adot, h, hdot, R]
    X0 = np.array([g/np.cos(gamma0), 0, hm0, -Vm_NED[2], 0])       # initial state vector

    # target initial conditions
    # randomly generated target initial conditions
    #ht0 = 1000 + 200*np.random.randn()
    ht0 = 1000 + 10+abs(50*np.random.randn())
    #ht0 = 950
    Vt = 200
    approach_angle = 90*Deg2Rad*(2*np.random.rand()-1)
    #approach_angle = np.pi/6
    psi0 = np.pi + approach_angle + 2*np.random.randn()*Deg2Rad
    #psi0 = np.pi*7/6
    psi0 = np.arctan2(np.sin(psi0), np.cos(psi0))

    Pt_N = 2000*(1+np.cos(approach_angle))
    Pt_E = 2000*np.sin(approach_angle)
    Pt_D = -ht0
    Pt_NED = np.array([Pt_N, Pt_E, Pt_D])                             # initial NED position
    Vt_NED = np.array([Vt*np.cos(psi0), Vt*np.sin(psi0), 0])       # initial NED velocity


    # initialize variables
    X = np.zeros((N,len(X0)))
    X[0,:] = X0
    dotX_p = 0

    Y = np.zeros((N,7))                       # print-out data
    theta0 = gamma0 + X0[0]*Acc2AoA + AoA0 # initial pitch angle

    DCM = np.zeros((3,3))                      # initial DCM NED-to-Body
    DCM[0,0] =  np.cos(theta0)
    DCM[0,2] = -np.sin(theta0)
    DCM[1,1] =  1
    DCM[2,0] =  np.sin(theta0)
    DCM[2,2] =  np.cos(theta0)

    Pr_NED = Pt_NED - Pm_NED                   # relative NED position
    Vr_NED = Vt_NED - Vm_NED                   # relative NED velosity

    Pr_Body = np.dot(DCM, Pr_NED)              # relative position (Body frame)

    # radar outputs
    r = np.linalg.norm(Pr_Body)                # range
    vc = -np.dot(Pr_NED, Vr_NED)/r             # closing velocity
    elev = np.arctan2(Pr_Body[2], Pr_Body[0])  # target vertival look angle (down +)
    azim = np.arctan2(Pr_Body[1], Pr_Body[0]/np.cos(theta0))  # target horizontal look angle (right +)

    los = theta0 - elev                        # line of sight angle
    dlos = 0
    daz = 0

    Y[0,:] = np.array([*Pm_NED, *Pt_NED,r]) 
    # static variables
    los_p = los
    dlos_p = dlos
    azim_p = azim
    daz_p = daz
    cmd_hold = False
    cmd_start = False
    direction_avoid = 0
    hdot_cmd = 0
    hdot = 0
    gamma = gamma0
    count_change_hdot=0
    count_change_hdot2=0
    count_vert_col=0
    err=0
    vc0=vc

    # main loop
    for k in range(N-1):  
        ##############################################################################
        # UPDATE ENVIRONMENT AND GET OBSERVATION

        # update environment
        # adams-bashforth 2nd order integration
        dotX = int_model(X[k,:], t[k], hdot_cmd)
        X[k+1,:] = X[k,:] + 0.5*(3*dotX-dotX_p)*dt
        dotX_p = dotX

        Pt_NED = Pt_NED + Vt_NED*dt        # target position integration

        # get observation

        a, adot, h, hdot, R = X[k+1,:]

        gamma = np.arcsin(hdot/Vm)
        theta = gamma + a*Acc2AoA + AoA0

        DCM = np.zeros((3,3))
        DCM[0,0] =  np.cos(theta)
        DCM[0,2] = -np.sin(theta)
        DCM[1,1] =  1
        DCM[2,0] =  np.sin(theta)
        DCM[2,2] =  np.cos(theta)

        Pm_NED = np.array([R, 0, -h]) 
        Vm_NED = np.array([Vm*np.cos(gamma), 0, -Vm*np.sin(gamma)])

        Pr_NED = Pt_NED - Pm_NED
        Vr_NED = Vt_NED - Vm_NED

        Pr_Body = np.dot(DCM, Pr_NED)

        r = np.linalg.norm(Pr_Body)
        vc = -np.dot(Pr_NED, Vr_NED)/r 
        elev = np.arctan2(Pr_Body[2], Pr_Body[0])
        azim = np.arctan2(Pr_Body[1], Pr_Body[0]/np.cos(theta))

        psi = np.arctan2(Vt_NED[1], Vt_NED[0])

        # los rate and az rate estimation
        los = theta - elev

        dlos = ( 30*(los-los_p) + 0*dlos_p ) / 3 # filtered LOS rate, F(s)=20s/(s+20)
        daz = ( 30*(azim-azim_p) + 0*daz_p ) / 3 # filtered azim rate, F(s)=20s/(s+20)

        los_p = los
        dlos_p = dlos
        azim_p = azim
        daz_p = daz

        # estimate closest approach
        min_dist_vert = r*r/vc*dlos
        min_dist_horiz = r*r/vc0*daz

        # estimate cruise distance
        dist_cruise = r*los

        ##############################################################################
        # COMPUTE ACTION (BEGIN)
        if k>3 and r>dist_sep and abs(elev)<40*Deg2Rad and abs(azim)<40*Deg2Rad:
            insight+=1
            data=torch.tensor(((np.array([r,vc,los,daz,dlos])
                 -mean)/std).astype(np.float32)).cuda()
            output=model(data.view(-1,5))
            _, predicted=torch.max(output,1)
            if predicted[0]==0:
                hdot_cmd=0
            if predicted[0]==1:
                if hdot_cmd!=-20:
                    count_change_hdot+=1
                hdot_cmd=-20
            if predicted[0]==2:
                if hdot_cmd!=20:
                    count_change_hdot+=1
                hdot_cmd=20
                



        ##############################################################################
        # WRITE DATA
        elif k>3:
            hdot_cmd=0
        Y[k+1,:] = np.array([*Pm_NED, *Pt_NED,r]) 
    if insight>0:
        hdot_res_cmd.append(count_change_hdot)
        res_Y[:,:,hdot_flag]=Y
        hdot_flag+=1
    if hdot_flag==total_sim:
        break

total_cor_mean=0

# In[ ]:


err=0
cor=0
cor_sum=0
disy=np.zeros(total_sim)
for i in range (total_sim):
    disy[i]=min(res_Y[:,6,i])
    if min(res_Y[:,6,i])<dist_sep:
        err+=1
    else:
        cor_sum+=min(res_Y[:,6,i])
        cor+=1
cor_mean=cor_sum/cor
total_cor_mean+=cor_mean

print("error with test down sim {}: ".format(total_sim), err)
print("Mean avoiding distance of correct avoidance with correction {}: ".format(cor), cor_mean)


# In[ ]:


plt.figure(figsize=(15,15))
sns.set(color_codes=True)
sns.distplot(disy)
plt.savefig(saving_img_path+"Down_"+model_char+".png", dpi=300)
plt.close()  


# In[ ]:

hdot_flag=0
res_Y = np.zeros(((N,7,total_sim)))                       # print-out data


hdot_res_cmd=[]
while True:
    hdot_res=[]
    hdot_res.append(0)
    time_temp=0
    errcmd=0
    insight=0
    
    
    hm0 = 1000                                                     # initial altitude
    Vm = 200                                                       # initial speed
    gamma0 = 0*Deg2Rad                                             # initial flight path angle
    Pm_NED = np.array([0, 0, -hm0])                                # initial NED position
    Vm_NED = np.array([Vm*np.cos(gamma0), 0, -Vm*np.sin(gamma0)])  # initial NED velocity

    # state variable: [a, adot, h, hdot, R]
    X0 = np.array([g/np.cos(gamma0), 0, hm0, -Vm_NED[2], 0])       # initial state vector

    # target initial conditions
    # randomly generated target initial conditions
    #ht0 = 1000 + 200*np.random.randn()
    ht0 = 1000 -10-abs(50*np.random.randn())
    #ht0 = 950
    Vt = 200
    approach_angle = 90*Deg2Rad*(2*np.random.rand()-1)
    #approach_angle = np.pi/6
    psi0 = np.pi + approach_angle + 2*np.random.randn()*Deg2Rad
    #psi0 = np.pi*7/6
    psi0 = np.arctan2(np.sin(psi0), np.cos(psi0))

    Pt_N = 2000*(1+np.cos(approach_angle))
    Pt_E = 2000*np.sin(approach_angle)
    Pt_D = -ht0
    Pt_NED = np.array([Pt_N, Pt_E, Pt_D])                             # initial NED position
    Vt_NED = np.array([Vt*np.cos(psi0), Vt*np.sin(psi0), 0])       # initial NED velocity


    # initialize variables
    X = np.zeros((N,len(X0)))
    X[0,:] = X0
    dotX_p = 0

    Y = np.zeros((N,7))                       # print-out data
    theta0 = gamma0 + X0[0]*Acc2AoA + AoA0 # initial pitch angle

    DCM = np.zeros((3,3))                      # initial DCM NED-to-Body
    DCM[0,0] =  np.cos(theta0)
    DCM[0,2] = -np.sin(theta0)
    DCM[1,1] =  1
    DCM[2,0] =  np.sin(theta0)
    DCM[2,2] =  np.cos(theta0)

    Pr_NED = Pt_NED - Pm_NED                   # relative NED position
    Vr_NED = Vt_NED - Vm_NED                   # relative NED velosity

    Pr_Body = np.dot(DCM, Pr_NED)              # relative position (Body frame)

    # radar outputs
    r = np.linalg.norm(Pr_Body)                # range
    vc = -np.dot(Pr_NED, Vr_NED)/r             # closing velocity
    elev = np.arctan2(Pr_Body[2], Pr_Body[0])  # target vertival look angle (down +)
    azim = np.arctan2(Pr_Body[1], Pr_Body[0]/np.cos(theta0))  # target horizontal look angle (right +)

    los = theta0 - elev                        # line of sight angle
    dlos = 0
    daz = 0

    Y[0,:] = np.array([*Pm_NED, *Pt_NED,r]) 
    # static variables
    los_p = los
    dlos_p = dlos
    azim_p = azim
    daz_p = daz
    cmd_hold = False
    cmd_start = False
    direction_avoid = 0
    hdot_cmd = 0
    hdot = 0
    gamma = gamma0
    count_change_hdot=0
    count_change_hdot2=0
    count_vert_col=0
    err=0
    vc0=vc

    # main loop
    for k in range(N-1):  
        ##############################################################################
        # UPDATE ENVIRONMENT AND GET OBSERVATION

        # update environment
        # adams-bashforth 2nd order integration
        dotX = int_model(X[k,:], t[k], hdot_cmd)
        X[k+1,:] = X[k,:] + 0.5*(3*dotX-dotX_p)*dt
        dotX_p = dotX

        Pt_NED = Pt_NED + Vt_NED*dt        # target position integration

        # get observation

        a, adot, h, hdot, R = X[k+1,:]

        gamma = np.arcsin(hdot/Vm)
        theta = gamma + a*Acc2AoA + AoA0

        DCM = np.zeros((3,3))
        DCM[0,0] =  np.cos(theta)
        DCM[0,2] = -np.sin(theta)
        DCM[1,1] =  1
        DCM[2,0] =  np.sin(theta)
        DCM[2,2] =  np.cos(theta)

        Pm_NED = np.array([R, 0, -h]) 
        Vm_NED = np.array([Vm*np.cos(gamma), 0, -Vm*np.sin(gamma)])

        Pr_NED = Pt_NED - Pm_NED
        Vr_NED = Vt_NED - Vm_NED

        Pr_Body = np.dot(DCM, Pr_NED)

        r = np.linalg.norm(Pr_Body)
        vc = -np.dot(Pr_NED, Vr_NED)/r 
        elev = np.arctan2(Pr_Body[2], Pr_Body[0])
        azim = np.arctan2(Pr_Body[1], Pr_Body[0]/np.cos(theta))

        psi = np.arctan2(Vt_NED[1], Vt_NED[0])

        # los rate and az rate estimation
        los = theta - elev

        dlos = ( 30*(los-los_p) + 0*dlos_p ) / 3 # filtered LOS rate, F(s)=20s/(s+20)
        daz = ( 30*(azim-azim_p) + 0*daz_p ) / 3 # filtered azim rate, F(s)=20s/(s+20)

        los_p = los
        dlos_p = dlos
        azim_p = azim
        daz_p = daz

        # estimate closest approach
        min_dist_vert = r*r/vc*dlos
        min_dist_horiz = r*r/vc0*daz

        # estimate cruise distance
        dist_cruise = r*los

        ##############################################################################
        # COMPUTE ACTION (BEGIN)
        if k>3 and r>dist_sep and abs(elev)<40*Deg2Rad and abs(azim)<40*Deg2Rad:
            insight+=1
            data=torch.tensor(((np.array([r,vc,los,daz,dlos])
                 -mean)/std).astype(np.float32)).cuda()
            output=model(data.view(-1,5))
            _, predicted=torch.max(output,1)
            if predicted[0]==0:
                hdot_cmd=0
            if predicted[0]==1:
                if hdot_cmd!=-20:
                    count_change_hdot+=1
                hdot_cmd=-20
            if predicted[0]==2:
                if hdot_cmd!=20:
                    count_change_hdot+=1
                hdot_cmd=20
                



        ##############################################################################
        # WRITE DATA
        elif k>3:
            hdot_cmd=0
        Y[k+1,:] = np.array([*Pm_NED, *Pt_NED,r]) 
    if insight>0:
        hdot_res_cmd.append(count_change_hdot)
        res_Y[:,:,hdot_flag]=Y
        hdot_flag+=1
    if hdot_flag==total_sim:
        break
        


# In[ ]:


err=0
cor=0
cor_sum=0
disy=np.zeros(total_sim)
for i in range (total_sim):
    disy[i]=min(res_Y[:,6,i])
    if min(res_Y[:,6,i])<dist_sep:
        err+=1
    else:
        cor_sum+=min(res_Y[:,6,i])
        cor+=1
cor_mean=cor_sum/cor
total_cor_mean+=cor_mean
total_cor_mean=total_cor_mean/2

print("error with test up sim {}: ".format(total_sim), err)
print("Mean avoiding distance of correct avoidance with correction {}: ".format(cor), cor_mean)

print("Mean avoiding distance both up and down: ", total_cor_mean)


# In[ ]:


plt.figure(figsize=(15,15))
sns.set(color_codes=True)
sns.distplot(disy)
plt.savefig(saving_img_path+"UP_"+model_char+".png", dpi=300)
plt.close()  


# In[ ]:


# player initial conditions

hdot_flag=0
res_Y = np.zeros(((N,7,total_sim)))                       # print-out data


hdot_res_cmd=[]
while True:
    hdot_res=[]
    hdot_res.append(0)
    time_temp=0
    errcmd=0
    insight=0
    
    
    hm0 = 1000                                                     # initial altitude
    Vm = 200                                                       # initial speed
    gamma0 = 0*Deg2Rad                                             # initial flight path angle
    Pm_NED = np.array([0, 0, -hm0])                                # initial NED position
    Vm_NED = np.array([Vm*np.cos(gamma0), 0, -Vm*np.sin(gamma0)])  # initial NED velocity

    # state variable: [a, adot, h, hdot, R]
    X0 = np.array([g/np.cos(gamma0), 0, hm0, -Vm_NED[2], 0])       # initial state vector

    # target initial conditions
    # randomly generated target initial conditions
    #ht0 = 1000 + 200*np.random.randn()
    if(random.choice([True, False])):
        ht0 = 1000 +120+10*np.random.randn()
    else:
        ht0 = 1000 -120-10*np.random.randn()
    #ht0 = 950
    Vt = 200
    approach_angle = 90*Deg2Rad*(2*np.random.rand()-1)
    #approach_angle = np.pi/6
    psi0 = np.pi + approach_angle + 2*np.random.randn()*Deg2Rad
    #psi0 = np.pi*7/6
    psi0 = np.arctan2(np.sin(psi0), np.cos(psi0))

    Pt_N = 2000*(1+np.cos(approach_angle))
    Pt_E = 2000*np.sin(approach_angle)
    Pt_D = -ht0
    Pt_NED = np.array([Pt_N, Pt_E, Pt_D])                             # initial NED position
    Vt_NED = np.array([Vt*np.cos(psi0), Vt*np.sin(psi0), 0])       # initial NED velocity


    # initialize variables
    X = np.zeros((N,len(X0)))
    X[0,:] = X0
    dotX_p = 0

    Y = np.zeros((N,7))                       # print-out data
    theta0 = gamma0 + X0[0]*Acc2AoA + AoA0 # initial pitch angle

    DCM = np.zeros((3,3))                      # initial DCM NED-to-Body
    DCM[0,0] =  np.cos(theta0)
    DCM[0,2] = -np.sin(theta0)
    DCM[1,1] =  1
    DCM[2,0] =  np.sin(theta0)
    DCM[2,2] =  np.cos(theta0)

    Pr_NED = Pt_NED - Pm_NED                   # relative NED position
    Vr_NED = Vt_NED - Vm_NED                   # relative NED velosity

    Pr_Body = np.dot(DCM, Pr_NED)              # relative position (Body frame)

    # radar outputs
    r = np.linalg.norm(Pr_Body)                # range
    vc = -np.dot(Pr_NED, Vr_NED)/r             # closing velocity
    elev = np.arctan2(Pr_Body[2], Pr_Body[0])  # target vertival look angle (down +)
    azim = np.arctan2(Pr_Body[1], Pr_Body[0]/np.cos(theta0))  # target horizontal look angle (right +)

    los = theta0 - elev                        # line of sight angle
    dlos = 0
    daz = 0

    Y[0,:] = np.array([*Pm_NED, *Pt_NED,r]) 
    # static variables
    los_p = los
    dlos_p = dlos
    azim_p = azim
    daz_p = daz
    cmd_hold = False
    cmd_start = False
    direction_avoid = 0
    hdot_cmd = 0
    hdot = 0
    gamma = gamma0
    count_change_hdot=0
    count_change_hdot2=0
    count_vert_col=0
    err=0
    vc0=vc

    # main loop
    for k in range(N-1):  
        ##############################################################################
        # UPDATE ENVIRONMENT AND GET OBSERVATION

        # update environment
        # adams-bashforth 2nd order integration
        dotX = int_model(X[k,:], t[k], hdot_cmd)
        X[k+1,:] = X[k,:] + 0.5*(3*dotX-dotX_p)*dt
        dotX_p = dotX

        Pt_NED = Pt_NED + Vt_NED*dt        # target position integration

        # get observation

        a, adot, h, hdot, R = X[k+1,:]

        gamma = np.arcsin(hdot/Vm)
        theta = gamma + a*Acc2AoA + AoA0

        DCM = np.zeros((3,3))
        DCM[0,0] =  np.cos(theta)
        DCM[0,2] = -np.sin(theta)
        DCM[1,1] =  1
        DCM[2,0] =  np.sin(theta)
        DCM[2,2] =  np.cos(theta)

        Pm_NED = np.array([R, 0, -h]) 
        Vm_NED = np.array([Vm*np.cos(gamma), 0, -Vm*np.sin(gamma)])

        Pr_NED = Pt_NED - Pm_NED
        Vr_NED = Vt_NED - Vm_NED

        Pr_Body = np.dot(DCM, Pr_NED)

        r = np.linalg.norm(Pr_Body)
        vc = -np.dot(Pr_NED, Vr_NED)/r 
        elev = np.arctan2(Pr_Body[2], Pr_Body[0])
        azim = np.arctan2(Pr_Body[1], Pr_Body[0]/np.cos(theta))

        psi = np.arctan2(Vt_NED[1], Vt_NED[0])

        # los rate and az rate estimation
        los = theta - elev

        dlos = ( 30*(los-los_p) + 0*dlos_p ) / 3 # filtered LOS rate, F(s)=20s/(s+20)
        daz = ( 30*(azim-azim_p) + 0*daz_p ) / 3 # filtered azim rate, F(s)=20s/(s+20)

        los_p = los
        dlos_p = dlos
        azim_p = azim
        daz_p = daz

        # estimate closest approach
        min_dist_vert = r*r/vc*dlos
        min_dist_horiz = r*r/vc0*daz

        # estimate cruise distance
        dist_cruise = r*los

        ##############################################################################
        # COMPUTE ACTION (BEGIN)
        if k>3 and r>dist_sep and abs(elev)<40*Deg2Rad and abs(azim)<40*Deg2Rad:
            insight+=1
            data=torch.tensor(((np.array([r,vc,los,daz,dlos])
                 -mean)/std).astype(np.float32)).cuda()
            output=model(data.view(-1,5))
            _, predicted=torch.max(output,1)
            if predicted[0]==0:
                hdot_cmd=0
            if predicted[0]==1:
                if hdot_cmd!=-20:
                    count_change_hdot+=1
                hdot_cmd=-20
            if predicted[0]==2:
                if hdot_cmd!=20:
                    count_change_hdot+=1
                hdot_cmd=20
                



        ##############################################################################
        # WRITE DATA
        elif k>3:
            hdot_cmd=0
        Y[k+1,:] = np.array([*Pm_NED, *Pt_NED,r]) 
    if insight>0:
        hdot_res_cmd.append(count_change_hdot)
        res_Y[:,:,hdot_flag]=Y
        hdot_flag+=1
    if hdot_flag==total_sim:
        break
        


# In[ ]:


err=0
disy=np.zeros(total_sim)
for i in range (total_sim):
    disy[i]=min(res_Y[:,6,i])
    if min(res_Y[:,6,i])<dist_sep:
        err+=1
print("error with test stay sim {}: ".format(total_sim), err)


# In[ ]:


plt.figure(figsize=(15,15))
sns.set(color_codes=True)
sns.distplot(disy)
plt.savefig(saving_img_path+"stay_"+model_char+".png", dpi=300)
plt.close()  

