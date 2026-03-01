import argparse
import os
import random
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
from collections import defaultdict
import time
import json

# カスタムモデルとデータセット、ユーティリティをインポート
# 劣化タイプ予測器として DegradationTypePredictor をインポート
from models.crop224_Res18_degradation_type_predictor import DegradationTypePredictor 
from datasets.ffhq import FFHQ as FFHQDataLoaderClass # データローダーヘルパークラス
import utils.plot_utils as plot_utils # 損失曲線プロット用

# ★ここから追加★
import matplotlib.pyplot as plt
import torchvision.transforms as T # 画像表示のための逆正規化など
# ★ここまで追加★


# configファイルをNamespaceオブジェクトに変換するヘルパー関数
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

# ★データ変換の逆変換関数 (表示用) ★
# モデルへの入力は[-1, 1]に変換されている場合があるため、0-1に戻す
def inverse_data_transform_display(X):
    # BCEWithLogitsLossを使う場合は、画像のデータ変換は0-1であるべき
    # もし画像が0-1に正規化されているだけならそのまま
    # もし [-1, 1] に変換されているなら、以下のように戻す
    # return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
    # 現在のffhq.pyのtransformsがToTensor()のみなので、0-1に収まっているはず。
    # ここではPIL.Image.openのRGB形式に合わせてRGBチャンネルの画像をそのまま使う
    return X 


def main():
    parser = argparse.ArgumentParser(description='Training Degradation Type Predictor (DTP) based on Liu et al. paper')
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the config file for DTP training")
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
    # DTPモデルの初期化
    print(f"=> creating Degradation Type Predictor (DTP) model...")
    dtp_model = DegradationTypePredictor(
        num_degradation_types=config.model.num_degradation_types
    ).to(device)

    # データ並列化 (必要であれば)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for DTP training.")
        dtp_model = nn.DataParallel(dtp_model)

    # オプティマイザ
    optimizer = getattr(optim, config.train.optimizer)(dtp_model.parameters(), lr=config.train.learning_rate)

    # 損失関数 (劣化種類分類のみ)
    # BCEWithLogitsLoss はマルチラベル分類や、ワンホットエンコーディングされた多クラス分類に適切
    type_criterion = nn.BCEWithLogitsLoss() 

    # -----------------------------------------------------
    # データローダーの準備
    print(f"=> loading dataset from '{config.data.data_dir}' for DTP training...")
    
    # FFHQDataLoaderClass が要求する config オブジェクトを調整 (既存のtrain_degradation_estimator.pyからコピー)
    mock_config_for_dataset = argparse.Namespace()
    mock_config_for_dataset.data = argparse.Namespace()
    mock_config_for_dataset.data.data_dir = config.data.data_dir
    mock_config_for_dataset.data.num_workers = config.data.num_workers
    # FFHQDatasetの__init__が要求するその他の引数にダミー値を渡す
    mock_config_for_dataset.training = argparse.Namespace(patch_n=1) 
    mock_config_for_dataset.data.image_size = 128 # これはFFHQDataset内のLRリサイズロジックに影響するので、適切な値であるべき

    mock_config_for_dataset.sampling = argparse.Namespace()
    mock_config_for_dataset.sampling.batch_size = config.sampling.batch_size

    ffhq_data_loader_helper = FFHQDataLoaderClass(mock_config_for_dataset)
    
    # 劣化推定器の訓練ではパッチ抽出は行わない (画像全体を見る)
    # FFHQDatasetの戻り値は (画像テンソル, img_id, deg_type_true, deg_params_true)
    train_dataset_loader, val_dataset_loader = ffhq_data_loader_helper.get_loaders(parse_patches=False)

    # -----------------------------------------------------
    # 訓練ループ
    print("\nStarting Degradation Type Predictor (DTP) training...")
    log_history = defaultdict(list)
    
    # チェックポイント保存フォルダ
    ckpt_dir = os.path.join(config.data.data_dir, 'dtp_checkpoints') # 新しいDTP用のフォルダ名
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(config.train.num_epochs):
        dtp_model.train() # 訓練モード
        total_train_type_loss = 0.0
        correct_predictions_train = 0
        total_samples_train = 0
        train_batch_count = 0
        epoch_start_time = time.time()

        for i, (x_combined, img_id_batch, deg_type_true, _) in enumerate(train_dataset_loader): 
            # 最適化
            optimizer.zero_grad()

            x_degraded_original_size = x_combined[:, :3, :, :] # 劣化した入力画像 (DTP入力前のサイズ)
            deg_type_true = deg_type_true.to(device)

            # DTPに画像を渡す (ここでクロップされる)
            x_degraded_for_dtp = x_degraded_original_size.to(device) # デバイスに転送
            type_logits_pred = dtp_model(x_degraded_for_dtp) 

            # 損失計算 (タイプ損失のみ)
            type_loss = type_criterion(type_logits_pred, deg_type_true)
            
            type_loss.backward() 
            optimizer.step()

            total_train_type_loss += type_loss.item()
            train_batch_count += 1

            # 精度計算 (訓練時)
            predicted_labels = (type_logits_pred > 0).float() # BCEWithLogitsLossの出力はロジットなので、0より大きいかで判断
            correct_predictions_train += (predicted_labels == deg_type_true).all(dim=1).sum().item() # 全ての要素が一致しているか (ワンホットなので)
            total_samples_train += deg_type_true.size(0)
            
            if True:#(i + 1) % config.train.log_freq == 0:
                avg_type_loss = total_train_type_loss / train_batch_count
                train_accuracy = (correct_predictions_train / total_samples_train) * 100
                #print(f"Epoch [{epoch}/{config.train.num_epochs}], Step [{i+1}/{len(train_dataset_loader)}], "
                     # f"Type Loss: {avg_type_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
                #log_history['train_type_loss'].append(type_loss.item()) 
                #log_history['train_accuracy'].append(train_accuracy) # ログに追加

                # ★ここから追加: デバッグ出力 (訓練時) ★
                if True:#(i+1) % (config.train.log_freq * 1) == 0: # 例として5倍の頻度で詳細出力
                    #print("\n--- Train Debug Output ---")
                    print(f"  Sample Image ID (Batch 0): {img_id_batch[0]}")
                    print(f"  Predicted Type Logits (Batch 0): {type_logits_pred[0].detach().cpu().numpy()}")
                    #print(f"  True Type One-Hot (Batch 0):     {deg_type_true[0].detach().cpu().numpy()}")
                    # 予測されたクラス (最も確率の高いカテゴリ)
                    predicted_class = torch.argmax(type_logits_pred[0]).item()
                    true_class = torch.argmax(deg_type_true[0]).item()
                    #print(f"  Predicted Class (Batch 0): {predicted_class}")
                    #print(f"  True Class (Batch 0):      {true_class}")
                    """
                    # 画像表示 (Batch 0の劣化入力画像)
                    # x_degraded_original_size はまだテンソル形式 (B, C, H, W)
                    img_to_display = inverse_data_transform_display(x_degraded_original_size[0].cpu()) # CPUに転送し、必要なら逆変換

                    plt.figure(figsize=(6, 6))
                    # PyTorchの画像テンソルは(C, H, W)なので、matplotlib表示のために(H, W, C)に変換
                    plt.imshow(img_to_display.permute(1, 2, 0).numpy())
                    plt.title(f"Train Image (ID: {img_id_batch[0]}) - True: {true_class}, Pred: {predicted_class}")
                    plt.axis('off')
                    plt.show(block=False) # block=False でノンブロッキング表示
                    plt.pause(4) # 短時間表示
                    plt.close() # ウィンドウを閉じる
                    """
                    print("--------------------------\n")
                # ★ここまで追加 ★

        # エポック終了時の平均損失
        avg_epoch_type_loss = total_train_type_loss / train_batch_count
        epoch_train_accuracy = (correct_predictions_train / total_samples_train) * 100
        log_history['train_type_loss'].append(avg_epoch_type_loss) 
        log_history['train_accuracy'].append(epoch_train_accuracy) # ログに追加
        print(f"Epoch {epoch} finished. Avg Type Loss: {avg_epoch_type_loss:.4f}, Avg Train Acc: {epoch_train_accuracy:.2f}%, "
              f"Time: {time.time() - epoch_start_time:.2f}s")
        
        # -----------------------------------------------------
        # 検証フェーズ
        dtp_model.eval() # 評価モード
        total_val_type_loss = 0.0
        correct_predictions_val = 0
        total_samples_val = 0
        val_batch_count = 0
        with torch.no_grad():
            for i, (x_combined, img_id_batch, deg_type_true, _) in enumerate(val_dataset_loader): 
                x_degraded_original_size = x_combined[:, :3, :, :]
                deg_type_true = deg_type_true.to(device)

                x_degraded_for_dtp = x_degraded_original_size.to(device) # デバイスに転送
                type_logits_pred = dtp_model(x_degraded_for_dtp)
                type_loss = type_criterion(type_logits_pred, deg_type_true)

                total_val_type_loss += type_loss.item()
                val_batch_count += 1

                # 精度計算 (検証時)
                predicted_labels = (type_logits_pred > 0).float()
                correct_predictions_val += (predicted_labels == deg_type_true).all(dim=1).sum().item()
                total_samples_val += deg_type_true.size(0)

                # ★ここから追加: デバッグ出力 (検証時) ★
                if True: #(i+1) % (config.train.log_freq // 10 if config.train.log_freq > 10 else 1) == 0: # 検証は頻繁に出力
                    #print("\n--- Validation Debug Output ---")
                    print(f"  Sample Image ID (Batch 0): {img_id_batch[0]}")
                    print(f"  Predicted Type Logits (Batch 0): {type_logits_pred[0].detach().cpu().numpy()}")
                    #print(f"  True Type One-Hot (Batch 0):     {deg_type_true[0].detach().cpu().numpy()}")
                    predicted_class = torch.argmax(type_logits_pred[0]).item()
                    true_class = torch.argmax(deg_type_true[0]).item()
                    #print(f"  Predicted Class (Batch 0): {predicted_class}")
                    #print(f"  True Class (Batch 0):      {true_class}")
                    """
                    # 画像表示 (Batch 0の劣化入力画像)
                    img_to_display = inverse_data_transform_display(x_degraded_original_size[0].cpu())

                    plt.figure(figsize=(6, 6))
                    plt.imshow(img_to_display.permute(1, 2, 0).numpy())
                    plt.title(f"Val Image (ID: {img_id_batch[0]}) - True: {true_class}, Pred: {predicted_class}")
                    plt.axis('off')
                    plt.show(block=False)
                    plt.pause(4)
                    plt.close()
                    """
                   # print("-------------------------------\n")
                # ★ここまで追加 ★
                
        avg_val_type_loss = total_val_type_loss / val_batch_count 
        val_accuracy = (correct_predictions_val / total_samples_val) * 100
        log_history['val_type_loss'].append(avg_val_type_loss)
        log_history['val_accuracy'].append(val_accuracy) # ログに追加
            
        print(f"Validation finished. Avg Val Type Loss: {avg_val_type_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

        # -----------------------------------------------------
        # チェックポイントの保存
        if (epoch + 1) % config.train.save_freq == 0 or epoch == config.train.num_epochs - 1:
            ckpt_path = os.path.join(ckpt_dir, f"dtp_epoch_{epoch+1}.pth") 
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': dtp_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_type_loss': log_history['train_type_loss'][-1] if log_history['train_type_loss'] else None,
                'val_type_loss': avg_val_type_loss,
                'train_accuracy': log_history['train_accuracy'][-1] if 'train_accuracy' in log_history and log_history['train_accuracy'] else None, # 追加
                'val_accuracy': val_accuracy # 追加                
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            
            # 損失履歴の保存 (JSON)
            with open(os.path.join(ckpt_dir, 'dtp_loss_history.json'), 'w') as f:
                json.dump(log_history, f, indent=4)
            
            # 損失曲線のプロット
            plot_utils.plot_loss(
                steps=np.arange(len(log_history['train_type_loss'])) * config.train.log_freq,
                losses=log_history['train_type_loss'],
                title='DTP Training Loss Curve',
                xlabel='Steps',
                ylabel='Loss',
                save_path=os.path.join(ckpt_dir, 'dtp_train_loss_curve.png')
            )
            plot_utils.plot_loss(
                steps=np.arange(len(log_history['val_type_loss'])) * config.train.save_freq * (len(train_dataset_loader)//config.train.log_freq if len(train_dataset_loader)//config.train.log_freq > 0 else 1),
                losses=log_history['val_type_loss'],
                title='DTP Validation Loss Curve',
                xlabel='Steps' if config.train.log_freq else 'Epochs',
                ylabel='Loss',
                save_path=os.path.join(ckpt_dir, 'dtp_val_loss_curve.png')
            )
            plot_utils.plot_multiple_losses(
                histories={
                    'Train Type Loss': (np.arange(len(log_history['train_type_loss'])) * config.train.log_freq, log_history['train_type_loss']),
                    'Validation Type Loss': (np.arange(len(log_history['val_type_loss'])) * config.train.save_freq * (len(train_dataset_loader)//config.train.log_freq if len(train_dataset_loader)//config.train.log_freq > 0 else 1), log_history['val_type_loss'])
                },
                title='DTP Train and Validation Loss',
                xlabel='Steps' if config.train.log_freq else 'Epochs',
                ylabel='Loss',
                save_path=os.path.join(ckpt_dir, 'dtp_combined_loss_curve.png')
            )
            # ★ここから追加: 精度曲線のプロット ★
            if 'train_accuracy' in log_history and log_history['train_accuracy']: # train_accuracyがある場合のみプロット
                 plot_utils.plot_loss(
                    steps=np.arange(len(log_history['train_accuracy'])) * config.train.log_freq,
                    losses=log_history['train_accuracy'],
                    title='DTP Training Accuracy Curve',
                    xlabel='Steps',
                    ylabel='Accuracy (%)',
                    save_path=os.path.join(ckpt_dir, 'dtp_train_accuracy_curve.png')
                 )
            if 'val_accuracy' in log_history and log_history['val_accuracy']: # val_accuracyがある場合のみプロット
                 plot_utils.plot_loss(
                    steps=np.arange(len(log_history['val_accuracy'])) * config.train.save_freq * (len(train_dataset_loader)//config.train.log_freq if len(train_dataset_loader)//config.train.log_freq > 0 else 1),
                    losses=log_history['val_accuracy'],
                    title='DTP Validation Accuracy Curve',
                    xlabel='Steps',
                    ylabel='Accuracy (%)',
                    save_path=os.path.join(ckpt_dir, 'dtp_val_accuracy_curve.png')
                 )
            # 精度曲線 (結合)
            if 'train_accuracy' in log_history and 'val_accuracy' in log_history and log_history['train_accuracy'] and log_history['val_accuracy']:
                plot_utils.plot_multiple_losses( # plot_lossではなくplot_multiple_losses
                    histories={
                        'Train Accuracy': (np.arange(len(log_history['train_accuracy'])) * config.train.log_freq, log_history['train_accuracy']),
                        'Validation Accuracy': (np.arange(len(log_history['val_accuracy'])) * config.train.save_freq * (len(train_dataset_loader)//config.train.log_freq if len(train_dataset_loader)//config.train.log_freq > 0 else 1), log_history['val_accuracy'])
                    },
                    title='DTP Train and Validation Accuracy',
                    xlabel='Steps' if config.train.log_freq else 'Epochs',
                    ylabel='Accuracy (%)',
                    save_path=os.path.join(ckpt_dir, 'dtp_combined_accuracy_curve.png')
                )
            # ★ここまで追加 ★


if __name__ == "__main__":
    main()