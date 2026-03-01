import torch
import numpy as np
import torch.nn as nn
import argparse
from transformers import CLIPTokenizer, CLIPTextModel, CLIPModel 

# -----------------------------------------------------
# CLIPモデルの初期化（事前にpip install transformers accelerateが必要）
# -----------------------------------------------------
# 注意: 環境によってはCLIPModelやProcessorのロードに時間がかかります。
# 訓練ループの外部で一度だけ初期化する必要があります。

# グローバル変数としてモデルとプロセッサを保持
_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_CLIP_TOKENIZER = None # プロセッサをトークナイザーに変更
CLIP_EMBEDDING_DIM = 512  # CLIP ViT/B-32などの一般的な次元
CLIP_MAX_LENGTH = 77

def initialize_clip(model_name="openai/clip-vit-base-patch32"):
    """CLIPモデルとトークナイザーを初期化する"""
    global _CLIP_MODEL, _CLIP_TOKENIZER
    if _CLIP_MODEL is None:
        print(f"Initializing CLIP model: {model_name}...")
        # ★変更: CLIPTextModelとCLIPTokenizerを直接ロード★
        # CLIPTextModelは、CLIPModelのtext_model部分
        _CLIP_MODEL = CLIPTextModel.from_pretrained(model_name)
        _CLIP_TOKENIZER = CLIPTokenizer.from_pretrained(model_name)
            
        _CLIP_MODEL.eval()
        print("CLIP initialization complete.")
    return _CLIP_MODEL, _CLIP_TOKENIZER

def degradation_to_text(deg_type_onehot: torch.Tensor, deg_params: torch.Tensor) -> str:
    """
    劣化タイプとパラメータのテンソルを受け取り、自然言語プロンプトに変換する
    
    Args:
        deg_type_onehot: 劣化タイプを示すワンホットベクトル (e.g., [1, 0, 0] for Noise)
        deg_params: 劣化パラメータのテンソル (e.g., [sigma, length, angle, scale])
    Returns:
        str: 劣化を説明する英語のプロンプト
    """
    # 簡略化のため、最も高い値のインデックスを取得
    deg_type_index = torch.argmax(deg_type_onehot).item()
    
    # パラメータ抽出 (正規化されていない生の値を想定)
    sigma = deg_params[0].item()
    length = deg_params[1].item()
    angle = deg_params[2].item()
    scale = deg_params[3].item()
    
    prompt = "A photo degraded by "

    if deg_type_index == 0: # Noise
        prompt += f"Gaussian noise with a standard deviation of {sigma:.1f}."
    elif deg_type_index == 1: # Motion Blur
        prompt += f"motion blur with a kernel length of {length:.1f} and an angle of {angle:.1f} degrees."
    elif deg_type_index == 2: # Downsampling/Low Resolution
        # ダウンサンプリングが適用されていると仮定
        prompt += f"downsampling to a scale factor of {scale:.2f}."
    elif deg_type_index == 3: # Other/Unknown
        prompt += "a complex or unknown degradation."
    else:
        prompt = "A clean, sharp photo." # 劣化がない場合
        
    return prompt

@torch.no_grad()
def get_clip_embedding(text_prompt: str, device: torch.device) -> torch.Tensor:
    """
    テキストプロンプトをCLIP埋め込み表現に変換する
    
    Args:
        text_prompt: 劣化を説明する自然言語プロンプト
        device: 実行デバイス (cuda or cpu)
    Returns:
        torch.Tensor: CLIP埋め込み表現 (1, Embedding_Dim)
    """
    model, processor = initialize_clip()
    model = model.to(device)

    # テキストをトークン化
    inputs = processor(text_prompt, return_tensors="pt", padding=True, truncation=True)
    
    # トークンをデバイスへ移動
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    # テキストエンコーダで埋め込みを生成
    # 埋め込みは通常、(1, Max_Length, Feature_Dim) の形状で返される
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # [CLS]トークンまたは最終層の出力を取得（CLIPの標準的な出力を想定）
    # CLIPのテキストエンコーダは通常、pooler_output (B, D) と last_hidden_state (B, L, D) を返す
    # 拡散モデルのクロスアテンションでは last_hidden_state が使われることが多い
    
    # ここでは、クロスアテンションに使用する last_hidden_state を返す
    # 形状は (1, 77, 768) (Max_Length=77, Feature_Dim=768)
    return output.last_hidden_state.squeeze(0)