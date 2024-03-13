import matplotlib.pyplot as plt
import numpy as np
from tensorboardX import SummaryWriter
loss_rcan_x2=[]
psnr_rcan_x2=[]
with open("log.txt")as f:
    files=f.readlines()
for i,x in enumerate(files):
    if((x.strip()!="") and (x.split()[0]=='[16000/16000]')):
        loss_rcan_x2.append(float(x.split()[2].split(']')[0]))
    if((x.strip()!="") and (x.split()[0]=='[DIV2K')):
        psnr_rcan_x2.append(float(x.split()[3].split(']')[0]))
f.close()
writer = SummaryWriter('/data1/home/zhangxiao/OISR-PyTorch-master/experiment/edsr_fde_x2_0.1_9')
for i,val in enumerate(psnr_rcan_x2):
    writer.add_scalar('psnr',val,i)
for i,val in enumerate(loss_rcan_x2):
    writer.add_scalar('loss',val,i)