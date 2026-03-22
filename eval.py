import os
import torch
import random
import argparse
import numpy as np
from models import *
from datasets import *
from config import _C as C
from torch.utils.data import DataLoader
from evaluator import PredEvaluator
import utils

def arg_parse():
    parser = argparse.ArgumentParser(description='Eval parameters')
    parser.add_argument('--cfg', required=True, help='path to config file', type=str)
    parser.add_argument('--ckpt', type=str, help='', default=None)
    parser.add_argument('--data-dir', type=str)
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
        print('[Warning] Fallback to CPU. Evaluation will be slower.')
        return torch.device('cpu')

    return torch.device('cuda')

def main():
    args = arg_parse()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = get_runtime_device()

    if device.type == 'cuda':
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(0)

    # --- setup config files
    C.merge_from_file(args.cfg)
    C.DATA_ROOT = utils.get_data_root()
    C.freeze()
    model_name = args.ckpt.split('/')[-2]
    iter_name = args.ckpt.split('/')[-1].split('.')[0]
    output_dir = os.path.join('eval_vis', model_name, iter_name+'-'+args.data_dir.replace('/','-'))

    # --- setup data loader
    dataset = eval(f'{C.DATASET_ABS}')(data_dir=os.path.join(C.DATA_ROOT, args.data_dir), phase='val')
    data_loader = DataLoader(dataset, batch_size=C.SOLVER.BATCH_SIZE, num_workers=0)

    model = dyn_model.Net()
    model.to(device)

    cp = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(cp['model'])
    tester = PredEvaluator(
        device=device,
        data_loader=data_loader,
        model=model,
        output_dir=output_dir,
    )
    tester.test()


if __name__ == '__main__':
    main()
