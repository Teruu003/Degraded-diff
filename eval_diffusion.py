import argparse
import os
import random
import socket
import yaml
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import models
import datasets
import utils
from models import DenoisingDiffusion, DiffusiveRestoration
# ★ここから追加★
from models.restoration import DiffusiveRestoration # restoration.py から DenoisingDiffusion クラスをインポート
# ★ここまで追加。もし restoration.py 内のクラス名が DenoisingDiffusion でない場合、
#   例えば RestorationEvaluator なら from models.restoration import RestorationEvaluator に変更し、
#   下の model = ... の行も合わせて変更してください。
#   以前の会話で models/restoration.py のクラスを DenoisingDiffusion として修正提案しているので、
#   このままで良い可能性が高いです。


def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Restoring Weather with Patch-Based Denoising Diffusion Models')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for the diffusion model checkpoint to load for evaluation')
    parser.add_argument("--grid_r", type=int, default=16,
                        help="Grid cell width r that defines the overlap between patches")
    parser.add_argument("--sampling_timesteps", type=int, default=25,
                        help="Number of implicit sampling steps")
    parser.add_argument("--test_set", type=str, default='raindrop',
                        help="restoration test set options: ['raindrop', 'snow', 'rainfog']")
    parser.add_argument("--image_folder", default='results/images/', type=str,
                        help="Location to save restored images")
    parser.add_argument('--seed', default=61, type=int, metavar='N',
                        help='Seed for initializing training (default: 61)')
    # ★ここから追加★
    parser.add_argument('--start_filename', default=None, type=str,
                        help='Start evaluation from a specific image file (e.g., "04012.png")')
    # ★ここまで追加★

    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def main():
    args, config = parse_args_and_config()

    # setup device to run
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device: {}".format(device))
    config.device = device

    if torch.cuda.is_available():
        print('Note: Currently supports evaluations (restoration) when run only on a single GPU!')

    # set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # data loading
    print("=> using dataset '{}'".format(config.data.dataset))
    DATASET = datasets.__dict__[config.data.dataset](config)
    _, val_loader = DATASET.get_loaders(parse_patches=False, validation=args.test_set)
    # ★ここから修正: 開始ファイル名からの処理ロジック★
    start_index = 0
    if args.start_filename:
        # val_loader.dataset は FFHQDataset のインスタンス
        # FFHQDataset の input_names に全ファイル名が含まれる
        try:
            # input_names はフルパスなので、ファイル名だけ比較するために os.path.basename を使う
            file_names_in_dataset = [os.path.basename(p) for p in val_loader.dataset.input_names]
            start_index = file_names_in_dataset.index(args.start_filename)
            print(f"Starting evaluation from {args.start_filename} at index {start_index}.")
        except ValueError:
            print(f"Warning: --start_filename {args.start_filename} not found in dataset. Starting from beginning.")
            start_index = 0
        
        # DataLoaderを再構築するか、Skipするイテレータを作成
        # ここでは、イテレータをラップしてスキップする
        # DataLoader の内部にアクセスして skip するのは難しいので、
        # Pythonの itertools.islice を使うのが簡単
        from itertools import islice
        val_loader_iter = islice(val_loader, start_index, None) # start_index から最後まで
    else:
        val_loader_iter = val_loader # 指定がなければ通常のローダーを使用
    # ★ここまで修正★

    # create model
    print("=> creating denoising-diffusion model with wrapper...")
    diffusion = DenoisingDiffusion(args, config)
    
    # モデルの初期化とチェックポイントのロード
    # 変更前: model = models.DenoisingDiffusion(diffusion,args, config)
    model = models.DiffusiveRestoration(diffusion, args, config) # ★ここを修正★    # モデルの復元（評価）を実行
    # val_loader_iter を渡す
    model.restore(val_loader_iter, validation=args.test_set,r=args.grid_r) # r は unused argument かもしれない



if __name__ == '__main__':
    main()
