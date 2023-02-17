#/bin/bash

wandb off
horovodrun -np 4 python train_cifar10_hvd.py 
