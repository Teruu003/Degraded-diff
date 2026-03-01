import os
from os import listdir
from os.path import isfile, join
import torch
import numpy as np
import torchvision
import torch.utils.data
import PIL.Image
import re
import random
import json # JSONを扱うためのインポート

import cv2 # OpenCVをインポート (グレースケール変換とローカルコントラスト正規化用)
#グレースケールバージョン　
#torchvision.transforms.ToTensor()で、ImageNetの平均と標準偏差で正規化バージョン。

# PIL.Image.Resampling の互換性対応 (generate_degraded_dataset.py と同様に)
try:
    _RESAMPLING_BICUBIC = PIL.Image.Resampling.BICUBIC
    _RESAMPLING_BILINEAR = PIL.Image.Resampling.BILINEAR
    _RESAMPLING_LANCZOS = PIL.Image.Resampling.LANCZOS # get_imagesでLANCZOSを使用しているため
except AttributeError:
    _RESAMPLING_BICUBIC = PIL.Image.BICUBIC
    _RESAMPLING_BILINEAR = PIL.Image.BILINEAR
    _RESAMPLING_LANCZOS = PIL.Image.LANCZOS

# ★ここから追加: ローカルコントラスト正規化の処理をPyTorchテンソルに適用するカスタムTransform ★
class LocalContrastNormalization(object):
    def __init__(self, window_size=11, constant=1e-6):
        # 論文の式(5)と(6)をTensorで実装
        self.window_size = window_size
        self.constant = constant
        
        # 畳み込みフィルターを定義 (平均を計算するため)
        self.filter_mean = nn.Conv2d(1, 1, kernel_size=window_size, padding=window_size//2, bias=False)
        self.filter_mean.weight.data.fill_(1.0 / (window_size**2))
        
        # 標準偏差を計算するためのフィルター (平均後の畳み込み)
        self.filter_std = nn.Conv2d(1, 1, kernel_size=window_size, padding=window_size//2, bias=False)
        self.filter_std.weight.data.fill_(1.0 / (window_size**2))

    def __call__(self, x):
        # x は(1, H, W)のグレースケールテンソル
        # PyTorchの畳み込みは(B, C, H, W)形式を期待するため、unsqueezeで次元を追加
        x = x.unsqueeze(0) # (1, 1, H, W)
        
        mu = self.filter_mean(x)
        mu_squared = self.filter_mean(x.pow(2))
        sigma = (mu_squared - mu.pow(2)).clamp(min=self.constant).sqrt()

        normalized_x = (x - mu) / (sigma + self.constant)
        
        # 元の次元に戻す
        return normalized_x.squeeze(0)

# ★ここまで追加 ★

class FFHQ:
    def __init__(self, config):
        self.config = config
        # ★ここを修正★: グレースケールに変換し、3チャンネルに戻す処理を削除
        self.transforms = torchvision.transforms.Compose([
            # 1. RGBをグレースケールに変換
            #torchvision.transforms.Grayscale(num_output_channels=1), # ★修正: 1チャンネルのグレースケールとして扱う★
            # 2. PyTorchテンソルに変換 (値は0-1)
            torchvision.transforms.ToTensor(),
            # 3. ImageNetの平均と標準偏差で正規化
            # グレースケール画像は1チャンネルなので、meanとstdも1チャンネルにする
            #torchvision.transforms.Normalize(mean=[0.449], std=[0.226]) # ★修正: グレースケール用の統計値を使用★
            # ※注: この統計値はImageNetのグレースケール版の標準的な値です。
        ])


    def get_loaders(self, parse_patches=True, validation='ffhq_test'):
        print(f"=> loading FFHQ paired dataset from '{self.config.data.data_dir}'...")

        # 訓練データセット
        train_dataset = FFHQDataset(
            dir=os.path.join(self.config.data.data_dir, 'train'),
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            filelist=None,
            parse_patches=parse_patches,
            # ★ 追加: degradation_labels_path を FFHQDataset に渡す ★
            degradation_labels_path=os.path.join(self.config.data.data_dir, 'degradation_labels.json')
        )

        # 検証データセット
        val_dataset = FFHQDataset(
            dir=os.path.join(self.config.data.data_dir, 'test'),
            n=self.config.training.patch_n,
            patch_size=self.config.data.image_size,
            transforms=self.transforms,
            filelist=None,
            parse_patches=parse_patches,
            # ★ 追加: degradation_labels_path を FFHQDataset に渡す ★
            degradation_labels_path=os.path.join(self.config.data.data_dir, 'degradation_labels.json')
        )
        """gpu不足を防ぐためのものらしい。
        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1
        """
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            self.config.training.batch_size,
            shuffle=True, # 訓練データはシャッフルする
            num_workers=self.config.data.num_workers,
            pin_memory=False
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False, # 検証データはシャッフルしない（固定順序）
            num_workers=self.config.data.num_workers,
            pin_memory=False
        )

        return train_loader, val_loader


class FFHQDataset(torch.utils.data.Dataset):
    # ★ 変更: degradation_labels_path 引数を追加 ★
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True, degradation_labels_path=None):
        super().__init__()

        self.dir = dir
        self.patch_size = patch_size
        self.n = n
        self.transforms = transforms
        self.parse_patches = parse_patches

        input_names, gt_names = [], []

        if filelist is None:
            input_sub_dir = os.path.join(dir, 'input')
            gt_sub_dir = os.path.join(dir, 'gt')

            if not os.path.isdir(input_sub_dir) or not os.path.isdir(gt_sub_dir):
                 raise FileNotFoundError(f"Expected '{input_sub_dir}' and '{gt_sub_dir}' directories for paired data. Please check your dataset structure.")

            images = [f for f in listdir(input_sub_dir) if isfile(join(input_sub_dir, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            if not images:
                 raise FileNotFoundError(f"No image files found in '{input_sub_dir}'. Please check your dataset structure.")

            input_names += [join(input_sub_dir, i) for i in images]
            gt_names += [join(gt_sub_dir, i) for i in images]

            input_names.sort()
            gt_names.sort()

            if len(input_names) != len(gt_names):
                raise ValueError(f"Mismatched number of input and ground truth images in '{dir}'. Input: {len(input_names)}, GT: {len(gt_names)}. Please ensure files are paired by name.")
            print(f"Found {len(input_names)} paired images in '{dir}'.")

        else:
            train_list = os.path.join(dir, filelist)
            with open(train_list) as f:
                contents = f.readlines()
                input_names = [i.strip() for i in contents]
                gt_names = [i.strip().replace('input', 'gt') for i in input_names]

            input_names.sort()
            gt_names.sort()
            if len(input_names) != len(gt_names):
                raise ValueError(f"Mismatched number of input and ground truth images based on filelist '{filelist}'. Input: {len(input_names)}, GT: {len(gt_names)}. Please ensure files are paired by name.")

        self.input_names = input_names
        self.gt_names = gt_names

        print(f"DEBUG: FFHQDataset initialized for dir: {dir}")
        print(f"DEBUG: Number of input_names: {len(self.input_names)}")
        if len(self.input_names) > 0:
            print(f"DEBUG: First input_name: {self.input_names[0]}")
            print(f"DEBUG: Last input_name: {self.input_names[-1]}")

        # ★ 追加: degradation_labels.json をロードする ★
        self.degradation_labels = self._load_degradation_labels(degradation_labels_path)
        print(f"DEBUG: Number of degradation_labels loaded: {len(self.degradation_labels)}")


    def _load_degradation_labels(self, label_filepath):
        """JSONファイルから劣化ラベルをロードする"""
        if not os.path.exists(label_filepath):
            raise FileNotFoundError(f"Degradation labels file not found at: {label_filepath}. Please run generate_degraded_dataset.py first.")
        with open(label_filepath, 'r') as f:
            labels = json.load(f)
        return labels

    @staticmethod
    def get_params(img, output_size, n):
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return [0] * n, [0] * n, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        return i_list, j_list, th, tw

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]

        img_id = re.split(r'[/\\]', input_name)[-1][:-4] # Windowsパスも考慮

        input_img = PIL.Image.open(input_name).convert('RGB')
        try:
            gt_img = PIL.Image.open(gt_name).convert('RGB')
        except:
            gt_img = PIL.Image.open(gt_name).convert('RGB')

        # ★ 追加: 劣化ラベルのロード ★
        file_base_name = os.path.basename(input_name) # '000000.png' のようなファイル名
        degradation_info = self.degradation_labels.get(file_base_name)

        if degradation_info is None:
            print(f"ERROR: Degradation labels not found for image: {file_base_name} in {self.degradation_labels_path}.")
            # このエラーが出た場合に、degradation_labels.json の内容や画像IDがどうなっているかを確認
            # print(f"DEBUG: Keys in degradation_labels: {list(self.degradation_labels.keys())[:10]} ...") # 最初の10個のキーを表示

            # ラベルが見つからない場合はエラーを発生させるか、デフォルト値を返す
            # ここではエラーを発生させ、ユーザーにファイル生成を確認させる
            raise ValueError(f"Degradation labels not found for image: {file_base_name} in {self.degradation_labels_path}. Please check your JSON file.")

        # Tensorに変換 (dtypeは必要に応じて調整)
        degradation_type_true = torch.tensor(degradation_info['type'], dtype=torch.float32)
        degradation_params_true = torch.tensor(degradation_info['params'], dtype=torch.float32)
        # ★ ここまで追加 ★

        if self.parse_patches:
            i, j, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), self.n)
            input_patches = self.n_random_crops(input_img, i, j, h, w)
            gt_patches = self.n_random_crops(gt_img, i, j, h, w)
            # ★ここから修正★
            # patch_n が 1 の場合は、スタックせずに直接テンソルを返す
            if self.n == 1:
                # 最初のパッチを取り出し、transformsを適用し、連結
                output_patch = torch.cat([self.transforms(input_patches[0]), self.transforms(gt_patches[0])], dim=0)
                return output_patch, img_id, degradation_type_true, degradation_params_true
            else:
                # patch_n が 1より大きい場合は、outputsのリストを作成しスタック
                outputs = [torch.cat([self.transforms(input_patches[k]), self.transforms(gt_patches[k])], dim=0)
                        for k in range(self.n)]
                return torch.stack(outputs, dim=0), img_id, degradation_type_true, degradation_params_true
            # ★ここまで修正★
        else:
            # Whole-image restoration (画像全体をそのまま利用)
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))

            input_img = input_img.resize((wd_new, ht_new), _RESAMPLING_LANCZOS)
            gt_img = gt_img.resize((wd_new, ht_new), _RESAMPLING_LANCZOS)

            # ★ 変更: 戻り値に劣化ラベルを追加 ★
            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id, degradation_type_true, degradation_params_true

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.input_names)