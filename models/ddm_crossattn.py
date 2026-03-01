import os
import time
import glob
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import utils
from models.unet import DiffusionUNet

# 追加するインポート
import matplotlib.pyplot as plt # グラフ描画用
import json # 辞書をJSON形式で保存するため
from collections import defaultdict # 辞書を初期化するため

# utils.plot_utils をインポートできるように、utilsフォルダのパスをPythonの検索パスに追加
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) # 親ディレクトリ (WeatherDiffusion-main/) をパスに追加
# utils.plot_utils をインポート
import utils.plot_utils as plot_utils
# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm


def data_transform(X):
    return 2 * X - 1.0


def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)


class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict


def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b, degradation_type=None, degradation_params=None):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    x = x0[:, 3:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()
    output = model(torch.cat([x0[:, :3, :, :], x], dim=1), t.float(), 
                   degradation_type=degradation_type, degradation_params=degradation_params)
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)



class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device

        self.model = DiffusionUNet(config)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        # --- ここから追加 ---
        self.loss_history = defaultdict(list) # 損失値を保存する辞書。train/loss, val/lossなどを区別可能
        self.log_dir = os.path.join(self.config.data.data_dir, 'logs', self.config.data.dataset)
        os.makedirs(self.log_dir, exist_ok=True) # ログディレクトリを作成
        print(f"Loss logs will be saved to: {self.log_dir}")
        # --- ここまで追加 ---


    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))


    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x_combined, img_id, degradation_type_true, degradation_params_true) in enumerate(train_loader):
                x = x_combined.flatten(start_dim=0, end_dim=1) if x_combined.ndim == 5 else x_combined
                n = x.size(0)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                x = data_transform(x)
                e = torch.randn_like(x[:, 3:, :, :])
                b = self.betas

                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                deg_type = degradation_type_true.to(self.device)  # (B, 3) ワンホット
                deg_params = degradation_params_true.to(self.device)  # (B, 4)
                
                loss = noise_estimation_loss(self.model, x, t, e, b, 
                                            degradation_type=deg_type,
                                            degradation_params=deg_params)

                # --- ここから追加 ---
                self.loss_history['train_loss'].append(loss.item()) # 損失値をリストに追加

                if self.step % self.config.training.logging_freq == 0:
                    # コンソール出力 (既存)
                    print(f"step: {self.step}, loss: {loss.item():.4f}, data time: {data_time / (i+1):.4f}")

                    # Loss履歴をJSONファイルに保存
                    loss_json_path = os.path.join(self.log_dir, 'loss_history.json')
                    with open(loss_json_path, 'w') as f:
                        json.dump(self.loss_history, f)
                    
                    # Loss曲線を画像ファイルとして保存
                    plot_utils.plot_loss(
                        steps=np.arange(len(self.loss_history['train_loss'])) * self.config.training.logging_freq, # x軸をステップ数に
                        losses=self.loss_history['train_loss'],
                        title='Training Loss Curve',
                        xlabel='Steps',
                        ylabel='Loss',
                        save_path=os.path.join(self.log_dir, 'train_loss_curve.png')
                    )
                # --- ここまで追加 ---
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                if self.step % self.config.training.validation_freq == 0:
                    self.model.eval()
                    self.sample_validation_patches(val_loader, self.step)

                if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    utils.logging.save_checkpoint({
                        'epoch': epoch + 1,
                        'step': self.step,
                        'state_dict': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'ema_helper': self.ema_helper.state_dict(),
                        'params': self.args,
                        'config': self.config
                    }, filename=os.path.join(self.config.data.data_dir, 'ckpts', self.config.data.dataset + '_ddpm'))

    def sample_image(self, x_cond, x, degradation_type=None, degradation_params=None, 
                    last=True, patch_locs=None, patch_size=None):
        # degradation_type: (B, 3) ワンホット
        # degradation_params: (B, 4) パラメータ
        # そのまま渡す
        
        skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps
        seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip)
        if patch_locs is not None:
            xs = utils.sampling.generalized_steps_overlapping(
                x, x_cond, seq, self.model, self.betas, eta=0.,
                degradation_type=degradation_type,
                degradation_params=degradation_params,
                corners=patch_locs, p_size=patch_size)
        else:
            xs = utils.sampling.generalized_steps(
                x, x_cond, seq, self.model, self.betas, eta=0.,
                degradation_type=degradation_type,
                degradation_params=degradation_params)
        if last:
            xs = xs[0][-1]
        return xs


    
    def sample_validation_patches(self, val_loader, step):
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset + str(self.config.data.image_size))
        
        # --- ここから追加 ---
        val_loss_sum = 0.0
        val_batch_count = 0
        # --- ここまで追加 ---

        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x_combined, img_id, degradation_type_true, degradation_params_true) in enumerate(val_loader):
                x = x_combined.flatten(start_dim=0, end_dim=1) if x_combined.ndim == 5 else x_combined
                n = x.size(0)
                
                # --- ここから追加 ---
                # 検証損失の計算
                x_val = x.to(self.device)
                x_val = data_transform(x_val)
                e_val = torch.randn_like(x_val[:, 3:, :, :])
                b_val = self.betas
                t_val = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t_val = torch.cat([t_val, self.num_timesteps - t_val - 1], dim=0)[:n]
                
                current_val_loss = noise_estimation_loss(self.model, x_val, t_val, e_val, b_val)
                val_loss_sum += current_val_loss.item()
                val_batch_count += 1
                # --- ここまで追加 ---

                # 元のコードの画像サンプリングはそのまま
                x_cond = x[:, :3, :, :].to(self.device)
                x_cond = data_transform(x_cond)
                x_sample = torch.randn(n, 3, self.config.data.image_size, self.config.data.image_size, device=self.device)
                x_sample = self.sample_image(x_cond, x_sample)
                x_sample = inverse_data_transform(x_sample)
                x_cond = inverse_data_transform(x_cond)

                for j in range(n):
                    utils.logging.save_image(x_cond[j], os.path.join(image_folder, str(step), f"{j}_cond.png"))
                    utils.logging.save_image(x_sample[j], os.path.join(image_folder, str(step), f"{j}.png"))
                
                # 検証画像を1バッチだけ処理するため break は維持
                break 
            
            # --- ここから追加 ---
            # 検証損失の平均を計算し、履歴に追加
            if val_batch_count > 0:
                avg_val_loss = val_loss_sum / val_batch_count
                self.loss_history['val_loss'].append(avg_val_loss)
                print(f"Validation Loss at step {step}: {avg_val_loss:.4f}")

                # 検証損失曲線も保存（train_loss_curve.png と同じファイルにまとめて描画しても良い）
                plot_utils.plot_loss(
                    steps=np.arange(len(self.loss_history['val_loss'])) * self.config.training.validation_freq,
                    losses=self.loss_history['val_loss'],
                    title='Validation Loss Curve',
                    xlabel='Steps',
                    ylabel='Loss',
                    save_path=os.path.join(self.log_dir, 'val_loss_curve.png')
                )
                # 両方の損失を一つのグラフに描画する例
                plot_utils.plot_multiple_losses(
                    histories={
                        'Train Loss': (np.arange(len(self.loss_history['train_loss'])) * self.config.training.logging_freq, self.loss_history['train_loss']),
                        'Validation Loss': (np.arange(len(self.loss_history['val_loss'])) * self.config.training.validation_freq, self.loss_history['val_loss'])
                    },
                    title='Train and Validation Loss',
                    xlabel='Steps',
                    ylabel='Loss',
                    save_path=os.path.join(self.log_dir, 'combined_loss_curve.png')
                )
            # --- ここまで追加 ---
