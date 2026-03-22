import os
import torch
import random
import argparse
import numpy as np
from datasets import *
from config import _C as cfg
from models import *
from trainer import Trainer
import utils

def arg_parse():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--cfg', required=True, help='config file', type=str)
    parser.add_argument('--init', type=str, help='init model from', default='')
    parser.add_argument('--exp-name', type=str, help='exp name')
    parser.add_argument('--seed', type=int, help='random seed', default=0)
    return parser.parse_args()


def get_runtime_device():
    if not torch.cuda.is_available():
        print('[Warning] CUDA is unavailable, fallback to CPU.')
        return torch.device('cpu')

    capability = torch.cuda.get_device_capability()
    arch = f'sm_{capability[0]}{capability[1]}'
    arch_list = torch.cuda.get_arch_list()
    if arch not in arch_list:
        print(f'[Warning] Current PyTorch does not support {arch} ({torch.cuda.get_device_name(0)}).')
        print('[Warning] Please reinstall a newer PyTorch build from https://pytorch.org/get-started/locally/')
        print('[Warning] Fallback to CPU. Training will be slower.')
        return torch.device('cpu')

    return torch.device('cuda')


def main():
    # ---- setup training environment
    args = arg_parse()
    rng_seed = args.seed
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    device = get_runtime_device()

    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)

    # ---- setup config files
    cfg.merge_from_file(args.cfg)
    cfg.DATA_ROOT = utils.get_data_root()
    cfg.freeze()

    # ---- setup model
    model = dyn_model.Net()
    model.to(device)

    # ---- setup optimizer
    optim = torch.optim.Adam(
        model.parameters(),
        lr=cfg.SOLVER.BASE_LR,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )

    # ---- if resume experiments, use --init ${model_name}
    if args.init:
        print(f'loading pretrained model from {args.init}')
        cp = torch.load(args.init)
        model.load_state_dict(cp['model'], False)

    # ---- setup dataset in the last, and avoid non-deterministic in data shuffling order
    random.seed(rng_seed)
    np.random.seed(rng_seed)
    torch.manual_seed(rng_seed)
    train_set = eval(f'{cfg.DATASET_ABS}')(data_dir=os.path.join(cfg.DATA_ROOT, cfg.TRAIN_DIR), phase='train')
    val_set = eval(f'{cfg.DATASET_ABS}')(data_dir=os.path.join(cfg.DATA_ROOT, cfg.VAL_DIR), phase='val')
    kwargs = {'pin_memory': False, 'num_workers': 4}

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, **kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, **kwargs,
    )
    print(f'size: train {len(train_loader)} / test {len(val_loader)}')

    # ---- setup trainer
    kwargs = {'device': device,
              'model': model,
              'optim': optim,
              'train_loader': train_loader,
              'val_loader': val_loader,
              'exp_name': args.exp_name,
              'max_iters': cfg.SOLVER.MAX_ITERS}
    trainer = Trainer(**kwargs)

    trainer.train()

if __name__ == '__main__':
    main()
