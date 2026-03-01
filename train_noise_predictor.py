import argparse
import os
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from collections import defaultdict
import time
import json

# モデルとデータセットをインポート
from models.noise_predictor import NoisePredictor
from datasets.ffhq import FFHQ as FFHQDataLoaderClass
import utils.plot_utils as plot_utils

# configファイルをNamespaceオブジェクトに変換
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
    parser = argparse.ArgumentParser(description='End-to-End Training of Noise Predictor')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file for noise predictor training")
    parser.add_argument('--resume', default='', type=str,
                        help='Path for checkpoint to load and resume')
    parser.add_argument('--seed', default=42, type=int,
                        help='Seed for initializing training (default: 42)')
    args = parser.parse_args()

    # configの読み込み
    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # シード設定
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    # -----------------------------------------------------
    # モデルの初期化
    # -----------------------------------------------------
    print("=> creating Noise Predictor model...")
    # FFHQDatasetはグレースケール1チャンネルを返す前提
    model = NoisePredictor(in_channels=1, out_channels=1, feature_channels=32).to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for model training.")
        model = nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=config.train.learning_rate)
    # ノイズレベル予測は回帰タスクなのでMSELossを使用
    loss_criterion = nn.MSELoss()

    # -----------------------------------------------------
    # データローダーの準備
    # -----------------------------------------------------
    mock_config_for_dataset = argparse.Namespace()
    mock_config_for_dataset.data = argparse.Namespace()
    mock_config_for_dataset.data.data_dir = config.data.data_dir
    mock_config_for_dataset.data.num_workers = config.data.num_workers
    mock_config_for_dataset.training = argparse.Namespace(patch_n=1) 
    mock_config_for_dataset.training.batch_size = config.train.batch_size
    mock_config_for_dataset.data.image_size = 128 
    mock_config_for_dataset.sampling = argparse.Namespace()
    mock_config_for_dataset.sampling.batch_size = config.sampling.batch_size

    ffhq_data_loader_helper = FFHQDataLoaderClass(mock_config_for_dataset)
    train_loader, val_loader = ffhq_data_loader_helper.get_loaders(parse_patches=False)

    log_history = defaultdict(list)
    ckpt_dir = os.path.join(config.data.data_dir, 'noise_predictor_ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # ★ここから追加★
    min_val_loss = float('inf') # 最小の検証ロスを追跡するための変数
    # ★ここまで追加★

    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if 'min_val_loss' in checkpoint:
                min_val_loss = checkpoint['min_val_loss']
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    print("\nStarting end-to-end training of Noise Predictor...")
    for epoch in range(start_epoch, config.train.num_epochs):
        model.train()
        total_train_loss = 0
        train_batch_count = 0

        for i, (x_combined, _, _, deg_params_true) in enumerate(train_loader):
            # 劣化画像と真のノイズ標準偏差を抽出
            noisy_image = x_combined[:, :1, :, :].to(device) # グレースケールなので1チャンネル
            true_sigma = deg_params_true[:, 0].to(device) # paramsの0番目がsigma

            optimizer.zero_grad()
            sigma_pred, _ = model(noisy_image)
            
            # 損失計算 (予測されたsigmaと真のsigmaの比較)
            loss = loss_criterion(sigma_pred.squeeze(), true_sigma)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batch_count += 1
            
            if (i + 1) % config.train.log_freq == 0:
                print(f"Epoch [{epoch}/{config.train.num_epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}, Avg.Loss: {total_train_loss / train_batch_count:.4f}")

        avg_epoch_loss = total_train_loss / train_batch_count
        log_history['train_loss'].append(avg_epoch_loss)
        print(f"Epoch {epoch} finished. Avg Loss = {avg_epoch_loss:.4f}")

        # 検証フェーズ
        model.eval()
        total_val_loss = 0
        val_batch_count = 0
        with torch.no_grad():
            for i, (x_combined, _, _, deg_params_true) in enumerate(val_loader):
                noisy_image = x_combined[:, :1, :, :].to(device)
                true_sigma = deg_params_true[:, 0].to(device)

                sigma_pred, _ = model(noisy_image)
                loss = loss_criterion(sigma_pred.squeeze(), true_sigma)

                total_val_loss += loss.item()
                val_batch_count += 1

        avg_val_loss = total_val_loss / val_batch_count
        log_history['val_loss'].append(avg_val_loss)
        print(f"Validation finished. Avg Val Loss = {avg_val_loss:.4f}")
    
        # ★ここから追加★
        # 最小ロスチェックポイントの保存
        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            best_ckpt_path = os.path.join(ckpt_dir, "estimator_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'min_val_loss': min_val_loss,
                #'train_loss': log_history['train_loss'][-1] if log_history['train_loss'] else None,
                'val_loss': avg_val_loss
            }, best_ckpt_path)
            print(f"New best checkpoint saved to {best_ckpt_path} with loss: {min_val_loss:.4f}")
        # ★ここまで追加★

        
        # チェックポイントの保存
        if (epoch + 1) % config.train.save_freq == 0 or epoch == config.train.num_epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"noise_predictor_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'min_val_loss': min_val_loss,
                'train_loss': avg_epoch_loss,
                'val_loss': avg_val_loss,
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            
            with open(os.path.join(ckpt_dir, 'loss_history.json'), 'w') as f:
                json.dump(log_history, f, indent=4)
            
            plot_utils.plot_multiple_losses(
                histories={
                    'Train Loss': (np.arange(len(log_history['train_loss'])), log_history['train_loss']),
                    'Validation Loss': (np.arange(len(log_history['val_loss'])), log_history['val_loss'])
                },
                title='Noise Predictor Loss Curves',
                xlabel='Epochs',
                ylabel='Loss (MSE)',
                save_path=os.path.join(ckpt_dir, 'loss_curves.png')
            )

if __name__ == "__main__":
    main()