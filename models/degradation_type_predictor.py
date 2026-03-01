import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF # クロップ処理のため
import random # ランダムな開始点を決めるため


# --- シンプルなCNNベースのDTP ---
class DegradationTypePredictor(nn.Module):
    def __init__(self, num_degradation_types=3):
        super().__init__()
        
        # 入力画像は3チャンネル (RGB)
        # 最初の畳み込み層
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1) # 32チャンネル出力
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # 画像サイズ半分 (例: 224->112)

        # 2番目の畳み込み層
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # 64チャンネル出力
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # 画像サイズ半分 (例: 112->56)

        # 3番目の畳み込み層
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1) # 128チャンネル出力
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # 画像サイズ半分 (例: 56->28)

        # 全結合層への入力サイズ計算
        # 入力サイズ224x224の場合:
        # conv1_out: (32, 224, 224)
        # pool1_out: (32, 112, 112)
        # conv2_out: (64, 112, 112)
        # pool2_out: (64, 56, 56)
        # conv3_out: (128, 56, 56)
        # pool3_out: (128, 28, 28)
        # 28 * 28 * 128 = 100352
        
        # ただし、入力画像サイズは 1024x1024 から始まるため、最終的な特徴マップのサイズは大きくなる
        # 例: 1024x1024 -> pool1 (512x512) -> pool2 (256x256) -> pool3 (128x128)
        # 128 * 128 * 128 = 2097152
        # このように大きな特徴マップになるので、最終的なGlobal Average Poolingを行う

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1)) # 空間次元を1x1に圧縮

        # 分類ヘッド
        self.classifier = nn.Linear(128, num_degradation_types) # Global Avg Pool後のチャンネル数

        # DTPの入力に必要なサイズ（FFHQが1024x1024なら、そのサイズを受け取る）
        # ただし、CNNは入力サイズに柔軟性があるため、厳密に指定しない
        # forwardでAdaptiveAvgPool2dが最終的な空間サイズを調整してくれる
        # このモデルは事前学習済みの重みを持たないため、全ての層が学習される。

        # DTPの入力に必要な固定サイズ
        self.target_input_size = 224 # DTPが常に224x224の画像を受け取るようにする

    def forward(self, x):
        # xは(B, 3, H, W) 劣化した入力画像
        
        # ランダムクロップは datasets/ffhq.py のFFHQDatasetで処理されるべき
        # もしFFHQDatasetの__getitem__が224x224のパッチを返しているなら、ここでリサイズは不要
        # しかし、現在のFFHQDatasetは1024x1024の画像から128x128パッチを抽出しているので、
        # このDTPの入力としてそのまま128x128が来る可能性が高い。
        # その場合、224x224にリサイズする代わりに、DTPの入力サイズに合わせてConv層の最終出力を調整するか、
        # AdaptiveAvgPool2dを活用する必要がある。

        # 論文のDTPは224x224にリサイズされている
        # SimpleCNNの場合も、入力サイズをある程度固定した方が良い
        # FFHQDatasetから来るパッチサイズ (self.config.data.image_size) がDTPの入力サイズになる
        # 例えば image_size: 128 なら DTPの入力は 128x128
        # 128x128 -> pool1 (64x64) -> pool2 (32x32) -> pool3 (16x16)
        # 16 * 16 * 128 = 32768
        
        # ここでは、FFHQDatasetが返す `self.config.data.image_size` （例: 128）を考慮して設計する
        # そのため、224x224へのF.interpolateは削除。
        # AdaptiveAvgPool2dが異なる入力サイズに対応するため、基本は不要。
        # しかし、もし入力サイズが極端に小さく、特徴マップが失われるようなら注意。

        # ★ここから修正: ランダムクロップのロジックを追加 ★
        processed_x_batch = []
        for i in range(x.size(0)): # バッチ内の各画像についてループ
            img_tensor = x[i] # 単一画像のテンソル (3, H, W)

            current_H, current_W = img_tensor.shape[1], img_tensor.shape[2]

            if current_H < self.target_input_size or current_W < self.target_input_size:
                # 画像がターゲットサイズより小さい場合、拡大してから中央クロップ
                # 低解像度劣化を見分ける目的と矛盾しないように注意が必要だが、
                # モデルの入力サイズを保証するためには必要
                print(f"Warning: Image {i} size {current_H}x{current_W} is smaller than target {self.target_input_size}. Upsampling then cropping.")
                img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(self.target_input_size, self.target_input_size), mode='bilinear', align_corners=False).squeeze(0)
                cropped_img = TF.center_crop(img_tensor, self.target_input_size)
                
            elif current_H == self.target_input_size and current_W == self.target_input_size:
                # すでにターゲットサイズの場合、そのまま使用
                cropped_img = img_tensor
            else:
                # 画像がターゲットサイズより大きい場合、ランダムクロップ
                # 例: 1024x1024 の画像から 224x224 を切り出す
                top = random.randint(0, current_H - self.target_input_size)
                left = random.randint(0, current_W - self.target_input_size)
                cropped_img = TF.crop(img_tensor, top, left, self.target_input_size, self.target_input_size)
            
            processed_x_batch.append(cropped_img)
        
        # 処理された単一の画像テンソルをバッチとしてスタック
        x_processed = torch.stack(processed_x_batch, dim=0)
        # ★修正ここまで ★

        h = self.pool1(self.relu1(self.conv1(x_processed)))
        h = self.pool2(self.relu2(self.conv2(h)))
        h = self.pool3(self.relu3(self.conv3(h)))

        # Global Average Pooling
        pooled_features = self.global_avg_pool(h).squeeze(-1).squeeze(-1) # (B, 128)

        # 分類ヘッド
        logits = self.classifier(pooled_features)
        
        return logits

# --- postprocess_degradation_params と DegradationEmbeddingNet は変更なし ---
# これらは models/degradation_type_predictor_simple_cnn.py と同じファイルに含めておく
# (以前の degradation_type_predictor.py のものと同じ)
def postprocess_degradation_params(type_logits, param_raw_values):
    predicted_type_idx = torch.argmax(type_logits, dim=1)
    filtered_params = torch.zeros_like(param_raw_values)
    for i in range(param_raw_values.shape[0]):
        if predicted_type_idx[i] == 0: # ノイズ
            filtered_params[i, 0] = param_raw_values[i, 0] 
        elif predicted_type_idx[i] == 1: # ブラー
            filtered_params[i, 1] = param_raw_values[i, 1]
            filtered_params[i, 2] = param_raw_values[i, 2]
        elif predicted_type_idx[i] == 2: # 低解像度
            filtered_params[i, 3] = param_raw_values[i, 3]
    return filtered_params

class DegradationEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim_type, embedding_dim_params, final_embedding_dim):
        super().__init__()
        self.type_embedding = nn.Linear(3, embedding_dim_type) 
        self.param_embedding = nn.Linear(4, embedding_dim_params)
        self.final_embedding_proj = nn.Linear(embedding_dim_type + embedding_dim_params, final_embedding_dim)
        
    def forward(self, type_one_hot_vector, filtered_params_vector):
        type_emb = self.type_embedding(type_one_hot_vector)
        param_emb = self.param_embedding(filtered_params_vector)
        
        combined_emb = torch.cat([type_emb, param_emb], dim=1)
        final_emb = self.final_embedding_proj(combined_emb)
        return final_emb