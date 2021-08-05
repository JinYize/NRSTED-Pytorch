import os
import torch
import torch.nn as nn
from datetime import datetime
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torchvision
from torchvision import *
from torch.utils.data import Dataset, DataLoader
from models import NRFED, NRFED_Dataset
from scipy.io import loadmat
from PIL import Image
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist


def train(gpu, args):
    ############################################################
    os.environ['MASTER_ADDR'] = 'localhost'
    rank = args.nr * args.gpus + gpu
    os.environ['MASTER_PORT'] = '888'+str(args.nr)
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )
    ############################################################

    torch.manual_seed(0)
    model = NRFED(1,0.2)
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 4
    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    ###############################################################
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model,
                                                device_ids=[gpu])
    ###############################################################

    ################################################################
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        args.train_dataset,
        num_replicas=args.world_size,
        rank=rank
    )
    ################################################################

    train_loader = torch.utils.data.DataLoader(
        dataset=args.train_dataset,
      batch_size=batch_size,
    ##############################
       shuffle=False,            #
    ##############################
       num_workers=0,
       pin_memory=True,
    ############################# 
      sampler=train_sampler)    # 
    #############################

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for itr, (x,y) in enumerate(train_loader):
            model.train()
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)
            output = model(x)

            loss = criterion(output,y)
            #
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (itr+1) % args.NDISP == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    itr + 1,
                    total_step,
                    loss.item())
                   )
                torch.save(model.state_dict(),'saved_models/model-'+str(epoch+1)+'_'+str(itr+1)+'.pt')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int,
                        metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--NDISP', default=100, type=int,
                        help='number of iterations to display loss info')
    parser.add_argument('--lr', default=1e-4, type=int,
                        help='learning rate')
    args = parser.parse_args()

    contents = []
    PATH = "databases/CSIQ_VQA/"
    text_file = open(PATH+"CSIQ_VQA_names.txt", "r")
    currContents = text_file.read().split('\n')
    for jj, eachName in enumerate(currContents[0:-1]):
        fullName = PATH+"fov_viewports/"+eachName
        contents.append(fullName)
    currFED = loadmat(PATH+"databases/CSIQ_VQA/CSIQ_VQA_name_srred.mat")["SRRED"]
    FEDScores = currFED

    args.train_dataset = NRFED_Dataset(contents,FEDScores)

    #########################################################
    args.world_size = args.gpus * args.nodes                #
    mp.spawn(train, nprocs=args.gpus, args=(args,))         #
    #########################################################

if __name__ == '__main__':
    main()
