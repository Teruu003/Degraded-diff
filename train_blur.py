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
from models.blur_predictor import VGG16_GrayScale_BlurPredictor # ここをBlurPredictorに修正
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
    print("=> creating Blur Predictor model...")
    # FFHQDatasetはグレースケール1チャンネルを返す前提
    model = VGG16_GrayScale_BlurPredictor(num_parameters=2).to(device)

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
    mock_config_for_dataset.data.image_size = 224 
    mock_config_for_dataset.sampling = argparse.Namespace()
    mock_config_for_dataset.sampling.batch_size = config.sampling.batch_size

    ffhq_data_loader_helper = FFHQDataLoaderClass(mock_config_for_dataset)
    train_loader, val_loader = ffhq_data_loader_helper.get_loaders(parse_patches=True)

    log_history = defaultdict(list)
    ckpt_dir = os.path.join(config.data.data_dir, 'blur_predictor_ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    min_val_loss = float('inf') # 最小の検証ロスを追跡するための変数

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

    # ブラーパラメータの正規化に使用する最大値と最小値を定義
    # ブラーの長さの範囲は [5, 25]
    length_min, length_max = 5.0, 25.0
    # ブラーの角度の範囲は [0, 180]
    angle_min, angle_max = 0.0, 180.0

    print("\nStarting end-to-end training of Blur Predictor...")
    for epoch in range(start_epoch, config.train.num_epochs):
        model.train()
        total_train_loss = 0
        total_train_loss_length = 0
        total_train_loss_angle = 0
        train_batch_count = 0

        for i, (x_combined, _, deg_type_true, deg_params_true) in enumerate(train_loader):
            # 劣化画像と真のブラーパラメータを抽出
            blurred_image = x_combined[:, :1, :, :].to(device) # グレースケールなので1チャンネル
            # ★ここから修正: 真のブラーパラメータを正規化する★
            true_length_unnormalized = deg_params_true[:, 1].to(device)
            true_angle_unnormalized = deg_params_true[:, 2].to(device)
            
            # 正規化: [min, max] -> [0, 1]
            true_length = (true_length_unnormalized - length_min) / (length_max - length_min)
            true_angle = (true_angle_unnormalized - angle_min) / (angle_max - angle_min)
            # ★ここまで修正★

            optimizer.zero_grad()
            length_pred, angle_pred = model(blurred_image)
            
            # 損失計算 (予測されたブラーパラメータと真のブラーパラメータの比較)
            loss_length = loss_criterion(length_pred.squeeze(), true_length)
            loss_angle = loss_criterion(angle_pred.squeeze(), true_angle)
            
            # 両方の損失を合計してバックプロパゲーション
            total_loss = loss_length + loss_angle

            total_loss.backward()
            optimizer.step()

            total_train_loss += total_loss.item()

            total_train_loss_length += loss_length.item()
            total_train_loss_angle += loss_angle.item()

            train_batch_count += 1
            
            if (i + 1) % config.train.log_freq == 0:
                avg_total_loss = (total_train_loss_length + total_train_loss_angle) / train_batch_count
                avg_length_loss = total_train_loss_length / train_batch_count
                avg_angle_loss = total_train_loss_angle / train_batch_count
                print(f"Epoch [{epoch}/{config.train.num_epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"Total Loss: {total_loss.item():.4f}, "
                      f"Avg. Total Loss: {avg_total_loss:.4f}, "
                      f"Avg. Length Loss: {avg_length_loss:.4f}, "
                      f"Avg. Angle Loss: {avg_angle_loss:.4f}")

        avg_epoch_loss_length = total_train_loss_length / train_batch_count
        avg_epoch_loss_angle = total_train_loss_angle / train_batch_count
        log_history['train_loss_length'].append(avg_epoch_loss_length)
        log_history['train_loss_angle'].append(avg_epoch_loss_angle)
        print(f"Epoch {epoch} finished. Avg Length Loss = {avg_epoch_loss_length:.4f}, Avg Angle Loss = {avg_epoch_loss_angle:.4f}")

        # 検証フェーズ
        model.eval()
        total_val_loss = 0
        total_val_loss_length = 0
        total_val_loss_angle = 0
        val_batch_count = 0
        with torch.no_grad():
            for i, (x_combined, _, deg_type_true, deg_params_true) in enumerate(val_loader):
                blurred_image = x_combined[:, :1, :, :].to(device)
                # ★ここから修正: 真のブラーパラメータを正規化する★
                true_length_unnormalized = deg_params_true[:, 1].to(device)
                true_angle_unnormalized = deg_params_true[:, 2].to(device)
                
                # 正規化: [min, max] -> [0, 1]
                true_length = (true_length_unnormalized - length_min) / (length_max - length_min)
                true_angle = (true_angle_unnormalized - angle_min) / (angle_max - angle_min)
                # ★ここまで修正★

                length_pred, angle_pred = model(blurred_image)
                loss_length = loss_criterion(length_pred.squeeze(), true_length)
                loss_angle = loss_criterion(angle_pred.squeeze(), true_angle)
                
                total_loss = loss_length + loss_angle

                total_val_loss += total_loss.item()
                total_val_loss_length += loss_length.item()
                total_val_loss_angle += loss_angle.item()
                val_batch_count += 1

        avg_val_loss = total_val_loss / val_batch_count
        avg_val_loss_length = total_val_loss_length / val_batch_count
        avg_val_loss_angle = total_val_loss_angle / val_batch_count
        log_history['val_loss_length'].append(avg_val_loss_length)
        log_history['val_loss_angle'].append(avg_val_loss_angle)
        print(f"Validation finished. Avg Val Length Loss = {avg_val_loss_length:.4f}, Avg Val Angle Loss = {avg_val_loss_angle:.4f}")

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
                'train_loss_length': avg_epoch_loss_length,
                'train_loss_angle': avg_epoch_loss_angle,
                'val_loss_length': avg_val_loss_length,
                'val_loss_angle': avg_val_loss_angle
            }, best_ckpt_path)
            print(f"New best checkpoint saved to {best_ckpt_path} with loss: {min_val_loss:.4f}")

        # チェックポイントの保存
        if (epoch + 1) % config.train.save_freq == 0 or epoch == config.train.num_epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"blur_predictor_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'min_val_loss': min_val_loss,
                'train_loss_length': avg_epoch_loss_length,
                'train_loss_angle': avg_epoch_loss_angle,
                'val_loss_length': avg_val_loss_length,
                'val_loss_angle': avg_val_loss_angle
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            
            with open(os.path.join(ckpt_dir, 'loss_history.json'), 'w') as f:
                json.dump(log_history, f, indent=4)
            
            # 損失曲線のプロット (両方のLossをプロット)
            plot_utils.plot_multiple_losses(
                histories={
                    'Train Length Loss': (np.arange(len(log_history['train_loss_length'])), log_history['train_loss_length']),
                    'Validation Length Loss': (np.arange(len(log_history['val_loss_length'])), log_history['val_loss_length']),
                    'Train Angle Loss': (np.arange(len(log_history['train_loss_angle'])), log_history['train_loss_angle']),
                    'Validation Angle Loss': (np.arange(len(log_history['val_loss_angle'])), log_history['val_loss_angle'])
                },
                title='Blur Predictor Loss Curves',
                xlabel='Epochs',
                ylabel='Loss (MSE)',
                save_path=os.path.join(ckpt_dir, 'loss_curves.png')
            )

if __name__ == "__main__":
    main()