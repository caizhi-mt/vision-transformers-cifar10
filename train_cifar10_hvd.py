# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''

from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pandas as pd
import csv
import time

from models import *
from utils import progress_bar
from randomaug import RandAugment
from models.vit import ViT
from models.convmixer import ConvMixer

import horovod.torch as hvd
from logger import create_logger
from timm.utils import accuracy, AverageMeter
import random
import pdb

# parsers
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=5e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
parser.add_argument('--opt', default="adam")
parser.add_argument('--resume', default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--resume_path', type=str, default='./OUTPUT/*.pth', help='resume file path from checkpoint')
parser.add_argument('--model_save_folder', default='./OUTPUT', type=str, help='folder used to save model')
parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
parser.add_argument('--nowandb', action='store_true', help='disable wandb')
parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
parser.add_argument('--net', default='swin')
parser.add_argument('--bs', default='512')
parser.add_argument('--size', default="32")
parser.add_argument('--device', default="mtgpu", type=str)
parser.add_argument('--n_epochs', type=int, default='400')
parser.add_argument('--seed', type=int, default='121')
parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
parser.add_argument('--dimhead', default="512", type=int)
parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

args = parser.parse_args()

# take in args
#usewandb = ~args.nowandb
#if usewandb:
#    import wandb
#    watermark = "{}_lr{}".format(args.net, args.lr)
#    wandb.init(project="cifar10-challange",
#            name=watermark, anonymous="allow")
#    wandb.config.update(args)

bs = int(args.bs)
imsize = int(args.size)

use_amp = not args.noamp

aug = args.noaug

device = args.device
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

hvd.init()
seed = args.seed + hvd.rank()
if args.device == "mtgpu":
    os.environ["PVR_GPUIDX"] = str(hvd.local_rank())
    os.environ["MTGPU_MAX_MEM_USAGE_GB"] = "31"
    import musa_torch_extension
if device == "cuda":
    torch.cuda.set_device(hvd.local_rank())
    cudnn.benchmark = True
    torch.cuda.manual_seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# Data
print('==> Preparing data..')
if args.net=="vit_timm":
    size = 384
else:
    size = imsize

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Add RandAugment with N, M(hyperparameter)
if aug:  
    N = 2; M = 14;
    transform_train.transforms.insert(0, RandAugment(N, M))
torch.set_num_threads(4)
# Prepare dataset
num_tasks = hvd.size()
global_rank = hvd.rank()

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
sampler_train = torch.utils.data.DistributedSampler(
    trainset, num_replicas=num_tasks, rank=global_rank, shuffle=True
)
trainloader = torch.utils.data.DataLoader(trainset, sampler=sampler_train, batch_size=bs, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
sampler_val = torch.utils.data.distributed.DistributedSampler(
    testset, num_replicas=num_tasks, rank=global_rank, shuffle=False
)
testloader = torch.utils.data.DataLoader(testset, sampler=sampler_val, batch_size=100, shuffle=False, num_workers=32)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model factory..
print('==> Building model..')
# net = VGG('VGG19')
if args.net=='res18':
    net = ResNet18()
elif args.net=='vgg':
    net = VGG('VGG19')
elif args.net=='res34':
    net = ResNet34()
elif args.net=='res50':
    net = ResNet50()
elif args.net=='res101':
    net = ResNet101()
elif args.net=="convmixer":
    # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
    net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
elif args.net=="mlpmixer":
    from models.mlpmixer import MLPMixer
    net = MLPMixer(
    image_size = 32,
    channels = 3,
    patch_size = args.patch,
    dim = 512,
    depth = 6,
    num_classes = 10
)
elif args.net=="vit_small":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_tiny":
    from models.vit_small import ViT
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 4,
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="simplevit":
    from models.simplevit import SimpleViT
    net = SimpleViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512
)
elif args.net=="vit":
    # ViT for cifar10
    net = ViT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)
elif args.net=="vit_timm":
    import timm
    net = timm.create_model("vit_base_patch16_384", pretrained=True)
    net.head = nn.Linear(net.head.in_features, 10)
elif args.net=="cait":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 8,
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="cait_small":
    from models.cait import CaiT
    net = CaiT(
    image_size = size,
    patch_size = args.patch,
    num_classes = 10,
    dim = int(args.dimhead),
    depth = 6,   # depth of transformer for patch to patch attention only
    cls_depth=2, # depth of cross attention of CLS tokens to patch
    heads = 6,
    mlp_dim = 256,
    dropout = 0.1,
    emb_dropout = 0.1,
    layer_dropout = 0.05
)
elif args.net=="swin":
    from models.swin import swin_t
    net = swin_t(window_size=args.patch,
                num_classes=10,
                downscaling_factors=(2,2,2,1))

net = net.to(device)
net_without_hvd = net
# For Multi-GPU
#if 'cuda' in device:
#    print(device)
#    print("using data parallel")
#    net = torch.nn.DataParallel(net) # make parallel
#    cudnn.benchmark = True

# Loss is CE
criterion = nn.CrossEntropyLoss()

if args.opt == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr)
elif args.opt == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr)  
    
optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())
# use cosine scheduling
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)

def load_checkpoint(model, resume_path, optimizer, lr_scheduler):
    checkpoint = torch.load(resume_path, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer'])
    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    start_epoch = checkpoint['epoch']

    del checkpoint
    return start_epoch
 

if args.resume:
    # Load checkpoint.
    print('==> Resume model paramter from ', args.resume_path)
    start_epoch = load_checkpoint(net, args.resume_path, optimizer, scheduler) + 1

hvd.broadcast_parameters(net.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)
##### Training
scaler = torch.cuda.amp.GradScaler(enabled=False)
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    optimizer.zero_grad()
    loss_meter = AverageMeter()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=False):
            outputs = net(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        optimizer.synchronize()
        has_nan = False
        for n, p in net.named_parameters():
            if p.grad is not None:
                if torch.any(torch.isnan(p.grad)):
                    has_nan = True
                    break
                if torch.any(p.grad > 20):
                    has_nan = True
                    break
                if has_nan:
                    optimizer.zero_grad()
        with optimizer.skip_synchronize():
            if not has_nan:
                scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        loss_meter.update(loss.item(), targets.size(0))
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (loss_meter.avg, 100.*correct/total, correct, total))
    return loss_meter.avg

##### Validation
def reduce_tensor_hvd(tensor):
    rt = tensor.clone()
    avg_rt = hvd.allreduce(rt)
    return avg_rt

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    acc_meter = AverageMeter()
    loss_meter = AverageMeter()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            acc = torch.tensor(correct/total)
            acc = reduce_tensor_hvd(acc)
            acc_meter.update(acc.item(), targets.size(0))
            loss = reduce_tensor_hvd(loss)
            loss_meter.update(loss.item(), targets.size(0))
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (loss_meter.avg, 100.*correct/total, correct, total))
    if hvd.rank() == 0:
        print(f' * Acc ', 100. * acc_meter.avg)
        print(f' * Average Loss ', loss_meter.avg)
    # Save checkpoint.
    #acc = 100.*correct/total
    #if acc > best_acc:
    #    print('Saving..')
    #    state = {"model": net.state_dict(),
    #          "optimizer": optimizer.state_dict(),
    #          "scaler": scaler.state_dict()}
    #    if not os.path.isdir('checkpoint'):
    #        os.mkdir('checkpoint')

    #    torch.save(state, './checkpoint/'+args.net+'-{}-ckpt.t7'.format(args.patch))
    #    best_acc = acc
    
    #os.makedirs("log", exist_ok=True)
    #content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
    #print(content)
    #with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
    #    appender.write(content + "\n")
    return loss_meter.sum, acc_meter.avg

list_loss = []
list_acc = []

#if usewandb:
#    wandb.watch(net)

def save_checkpoint(epoch, model, max_accuracy, optimizer, lr_scheduler):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'epoch': epoch}
    save_path = os.path.join(args.model_save_folder, f'ckpt_epoch_{epoch}.pth')
    torch.save(save_state, save_path)

for epoch in range(start_epoch, args.n_epochs):
    start = time.time()
    trainloader.sampler.set_epoch(epoch)
    trainloss = train(epoch)
    if hvd.rank() == 0 and (epoch % 100 == 50):
        save_checkpoint(epoch, net_without_hvd, 0.0, optimizer, scheduler)
    val_loss, acc = test(epoch)
    
    scheduler.step(epoch-1) # step cosine scheduling
    
    list_loss.append(val_loss)
    list_acc.append(acc)
    
    # Log training..
    #if usewandb:
    #    wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
    #    "epoch_time": time.time()-start})

    # Write out csv..
    with open(f'log/log_{args.net}_patch{args.patch}.csv', 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(list_loss) 
        writer.writerow(list_acc) 
    #print(list_loss)

# writeout wandb
#if usewandb:
#    wandb.save("wandb_{}.h5".format(args.net))
    
