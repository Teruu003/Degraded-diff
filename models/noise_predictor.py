import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
#エンドツーエンドで学習できるようにした。

# --- 1. ノイズセパレータ (浅いCNN) ---
# 論文のFig. 2と2.3節の記述に基づく
class NoiseSeparator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_channels=32):
        super().__init__()
        # 論文では3チャンネルを使用していますが、ここではグレースケール（1チャンネル）を使用します。
        self.conv1 = nn.Conv2d(in_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(feature_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(feature_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv_out = nn.Conv2d(feature_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, noisy_image):
        """
        Args:
            noisy_image (torch.Tensor): 入力ノイズ画像 (B, C, H, W)。
        Returns:
            torch.Tensor: 推定されたノイズ成分 (B, C, H, W)。
        """
        h = self.relu1(self.bn1(self.conv1(noisy_image)))
        h = self.relu2(self.bn2(self.conv2(h)))
        noise_component = self.conv_out(h)
        return noise_component


# --- 2. GGDパラメータ推定器 (統計的計算をNNに置き換え) ---
# このモジュールが、ノイズ成分からGGDパラメータを予測する。
# これにより、パイプライン全体が微分可能になる。
class GGDParameterEstimator(nn.Module):
    # ★修正: in_channels を 1 に設定し、MLPの最初の層を修正★
    def __init__(self, in_channels, num_parameters=2):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # MLPの入力はin_channelsに合わせる
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 32), # ★in_channelsが1に設定される
            nn.ReLU(),
            nn.Linear(32, num_parameters)
        )

    def forward(self, noise_component):
        """
        Args:
            noise_component (torch.Tensor): 推定されたノイズ成分 (B, C, H, W)。
        Returns:
            torch.Tensor: 推定されたGGDパラメータ (α, β) (B, 2)。
        """
        # グローバル平均プーリング
        pooled_features = self.avg_pool(noise_component) # (B, C, 1, 1)
        # フラット化
        flattened_features = pooled_features.view(pooled_features.size(0), -1) # (B, C)
        # MLPでGGDパラメータを予測
        ggd_params = self.mlp(flattened_features) # (B, 2)
        return ggd_params

# --- 3. ノイズレベルマッピング (BPニューラルネットワーク) ---
# 論文のFig. 2と2.5節に基づく
class BPNoiseLevelMapper(nn.Module):
    def __init__(self):
        super().__init__()
        # 2つの隠れ層、各6ニューロン。入力は2つ (alpha, beta)、出力は1つ (sigma)。
        self.fc1 = nn.Linear(2, 6)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(6, 6)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc3 = nn.Linear(6, 1)

    def forward(self, features):
        """
        Args:
            features (torch.Tensor): GGDパラメータ (alpha, beta) を結合した特徴ベクトル (B, 2)。
        Returns:
            torch.Tensor: 推定されたノイズレベルsigma (B, 1)。
        """
        h = self.relu1(self.fc1(features))
        h = self.relu2(self.fc2(h))
        sigma_pred = self.fc3(h)
        return sigma_pred

# --- 3つのモジュールを組み合わせた完全なノイズ予測モデル ---
class NoisePredictor(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, feature_channels=32):
        super().__init__()
        self.noise_separator = NoiseSeparator(in_channels, out_channels, feature_channels)
        
        # ★ここを修正★: GGDParameterEstimatorのin_channelsをout_channelsに合わせる
        self.ggd_estimator = GGDParameterEstimator(in_channels=out_channels) # <-- 修正
        
        self.bp_mapper = BPNoiseLevelMapper()

    def forward(self, noisy_image):
        """
        ノイズ画像からノイズレベルを予測するエンドツーエンドのフォワードパス。
        Args:
            noisy_image (torch.Tensor): 入力ノイズ画像 (B, C, H, W)。
        Returns:
            tuple: (sigma_pred, ggd_params)
        """
        # 1. ノイズセパレータ
        noise_component = self.noise_separator(noisy_image)
        
        # 2. GGDパラメータ推定器 (NNベース)
        ggd_params = self.ggd_estimator(noise_component)
        
        # 3. ノイズレベルマッピング (BPネットワーク)
        sigma_pred = self.bp_mapper(ggd_params)

        return sigma_pred, ggd_params # 両方の出力を返すことで、損失計算に利用可能