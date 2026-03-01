import torch
import torch.nn as nn
import torch.nn.functional as F

class VGG16_GrayScale_BlurPredictor(nn.Module):
    """
    VGG16のアーキテクチャに基づき、グレースケール（1チャンネル）入力に対応したモデル。
    事前学習済み重みは使用せず、スクラッチから学習します。
    """
    def __init__(self, num_parameters=2):
        super().__init__()
        
        # VGG16のfeatures部分のアーキテクチャを1チャンネル入力で再現
        # 最初の層のin_channelsを1に設定
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # AvgPool層
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        # 回帰ヘッド（論文のVGG16_Regressor.pyを参考に）
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5), # 過学習対策
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_parameters), # 論文の通り、長さと角度の2つの出力を予測
            nn.Sigmoid() # 論文の通り、出力を [0, 1] に正規化するためシグモイドを使用
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 入力画像テンソル (B, 1, H, W)。
        Returns:
            tuple: (length_pred, angle_pred)
        """
        # 特徴抽出
        x = self.features(x)
        
        # 平均プーリング
        x = self.avgpool(x)
        
        # フラット化
        x = torch.flatten(x, 1)
        
        # 回帰ヘッド
        predictions = self.classifier(x)
        
        # 長さと角度の予測値 (B, 2)
        length_pred = predictions[:, 0:1] # (B, 1)
        angle_pred = predictions[:, 1:2]  # (B, 1)
        
        return length_pred, angle_pred
