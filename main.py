import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from settings import cfg
import torch
from multiprocessing import freeze_support
from trainer import TrainerV0


def main():
    trainer = TrainerV0(cfg)
    trainer.train()


if __name__ == "__main__":
    torch.manual_seed(1214090112858600)
    torch.cuda.manual_seed(295516382103593)
    freeze_support()
    main()