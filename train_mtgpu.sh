#/bin/bash
export PVR_GPUIDX=7
export MTGPU_MAX_MEM_USAGE_GB=32
wandb off
python train_cifar10.py --net swin --n_epochs 400 --noamp --device mtgpu
