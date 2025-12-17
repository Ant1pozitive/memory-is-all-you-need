import argparse
import os
from config import cfg
from train import run_experiment

def make_dir(path):
    os.makedirs(path, exist_ok=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='default')
    parser.add_argument('--task', type=str, default='copy')
    args = parser.parse_args()

    exp_root = os.path.join('experiments', args.exp)
    make_dir(exp_root)
    run_experiment(args.task, exp_root)
