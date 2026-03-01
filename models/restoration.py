import torch
import torch.nn as nn
import utils
import torchvision
import os
import math

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class DiffusiveRestoration:
    def __init__(self, diffusion, args, config):
        super(DiffusiveRestoration, self).__init__()
        self.args = args
        self.config = config
        self.diffusion = diffusion

        if os.path.isfile(args.resume):
            self.diffusion.load_ddm_ckpt(args.resume, ema=True)
            self.diffusion.model.eval()
        else:
            print('Pre-trained diffusion model path is missing!')

    def restore(self, val_loader, validation='raindrop', r=None):
        """
        検証データローダーから画像を読み込み、復元を行い、保存する。
        """
        image_folder = os.path.join(self.args.image_folder, self.config.data.dataset, validation)
        os.makedirs(image_folder, exist_ok=True)
        
        with torch.no_grad():
            for i, (x_combined, img_id, degradation_type_true, degradation_params_true) in enumerate(val_loader):
                # img_id の処理 (タプルやリストの場合があるため)
                if isinstance(img_id, (list, tuple)):
                    current_img_id = img_id[0]
                else:
                    current_img_id = img_id

                print(f"Processing image: {current_img_id}")

                # x_combined: (B, 6, H, W) -> 学習時と同じ結合データ
                # 検証時は batch_size=1 を想定
                x = x_combined
                if x.ndim == 5: # パッチ化されている場合 (B, P, 6, H, W) -> フラット化
                     x = x.flatten(start_dim=0, end_dim=1)

                # 条件画像 (劣化画像) を抽出 (Ch 0-2)
                x_cond = x[:, :3, :, :].to(self.diffusion.device)
                x_cond = data_transform(x_cond) # [-1, 1] に正規化

                # 劣化パラメータの準備
                # (B, 4) のテンソルにしてGPUへ
                deg_params = self.diffusion._degradation_params_to_tensor(degradation_params_true).to(self.diffusion.device)

                # 復元実行 (オーバーラップサンプリング)
                # r: グリッドのストライド (Noneならconfigのデフォルトか16)
                x_output = self.diffusive_restoration(x_cond, deg_params, r=r)

                # 保存用に逆変換
                x_output = inverse_data_transform(x_output)
                
                # ファイル名生成
                y_filename = os.path.splitext(os.path.basename(current_img_id))[0]
                output_path = os.path.join(image_folder, f"{y_filename}.png")
                
                print(f"Saving to: {output_path}")
                utils.logging.save_image(x_output, output_path)

    def diffusive_restoration(self, x_cond, degradation_params, r=None):
        """
        オーバーラップサンプリングを用いて画像を復元する。
        """
        # パッチサイズ (学習時のサイズ、例: 64)
        p_size = self.config.data.image_size
        
        # グリッド座標の計算 (左上の角の座標リスト)
        h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=p_size, r=r)
        corners = [(i, j) for i in h_list for j in w_list]
        
        # 初期ノイズ生成 (入力画像全体と同じサイズ)
        x = torch.randn(x_cond.size(), device=self.diffusion.device)
        
        # サンプリング実行
        # patch_locs (corners) を渡すことで、generalized_steps_overlapping が呼ばれる
        x_output = self.diffusion.sample_image(
            x_cond, 
            x, 
            degradation_params=degradation_params, # AdaGN用パラメータ
            patch_locs=corners, 
            patch_size=p_size
        )
        return x_output

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        """
        画像全体をカバーするオーバーラップグリッドのインデックスを生成する。
        """
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r # ストライド (重なり具合を制御)
        
        # グリッド生成
        h_list = [i for i in range(0, h - output_size + 1, r)]
        w_list = [i for i in range(0, w - output_size + 1, r)]
        
        # 右端と下端がカバーされていない場合、最後のパッチを追加する
        if h_list[-1] + output_size < h:
            h_list.append(h - output_size)
        if w_list[-1] + output_size < w:
            w_list.append(w - output_size)
            
        return h_list, w_list