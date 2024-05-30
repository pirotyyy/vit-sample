import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from model import VisionTransformer
from tqdm import tqdm
from dataset import load_data
from config import load_config
from omegaconf import DictConfig, OmegaConf
from vit_pytorch.efficient import ViT

from linformer import Linformer


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train(conf):
    # load conf
    print('Use Following Config: \n', conf)

    # device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # set seed
    seed_everything(conf['seed'])

    # load_data
    train_loader, val_loader, test_loader, _ = load_data(conf['batch_size'])

    efficient_transformer = Linformer(
        dim=128,
        seq_len=49+1,
        depth=12,
        heads=8,
        k=64
    )

    model = ViT(
        dim=128,
        image_size=224,
        patch_size=32,
        num_classes=10,
        transformer=efficient_transformer,
        channels=3
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=conf['lr'])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=conf['gamma'])

    



def main():
    conf = load_config()
    train(conf)

    __ = load_data(100)


if __name__ == '__main__':
    main()