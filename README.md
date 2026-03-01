# Degraded-diff — Degradation-aware Conditional Diffusion for Face Image Restoration

劣化情報（劣化タイプ + 劣化パラメータ）を推定し、拡散モデルの復元過程をガイドすることで、劣化顔画像の復元精度を向上させる研究実装です。

- 劣化推定：劣化タイプ分類 / ノイズσ推定 / モーションブラー（length, angle）推定
- 条件付け：推定結果をテキスト化 → CLIP(ViT-B/32)で埋め込み → Cross-Attentionで拡散モデルへ統合
- 評価：PSNR / SSIM（w/o degradation info との比較）

> 就活向け要約：不確実な劣化を「推定→条件付け→生成モデルで復元」に分解し、Pythonで 設計→実装→定量評価→改善 を回した研究開発です。

---

## Repository Structure

- `configs/ffhq.yml` : diffusion 学習・復元用 config（※実在）
- `train_diffusion.py` : diffusion model の学習（`configs/<name>.yml` を読み込む）
- `eval_diffusion.py` : 復元（推論/評価）
- `train_degradation_type_predictor.py` : 劣化タイプ分類器の学習
- `train_noise_predictor.py` : ノイズσ推定器の学習
- `train_blur.py` : ブラー(length, angle)推定器の学習
- `datasets/` / `models/` / `utils/` : dataloader / model / utilities

---

## Quick Start（最短で動かす）

### 0) Install

```bash
pip install torch torchvision pyyaml numpy opencv-python matplotlib
```

### 1) Configの確認（重要）

Diffusion用 config は `configs/ffhq.yml` を使います。  
この config にはデータセットパスが **Windowsの絶対パス**で書かれているため、必ず自分の環境に合わせて変更してください。

- `configs/ffhq.yml` の `data.data_dir` を変更（例：`./datasets/data/ffhq` など）
- 現状の例：`C:/Users/.../datasets/data/ffhq`

### 2) Train Diffusion

```bash
python train_diffusion.py --config ffhq.yml
```

### 3) Restore / Evaluate

（diffusionのチェックポイントを指定）

```bash
python eval_diffusion.py --config ffhq.yml --resume <PATH_TO_DIFFUSION_CKPT> --test_set ffhq
```

---

## 推定器（Degradation Estimators）

### 重要：推定器は専用configが必要です

`train_noise_predictor.py` は `config.train.learning_rate`, `config.train.batch_size` など **`train:` セクション**を参照します。  
一方、`configs/ffhq.yml` 側は `training:` なのでそのままでは一致しません。

そこで、以下の YAML を **`configs/noise_predictor.yml` / `configs/degradation_type_predictor.yml` / `configs/blur_predictor.yml`** として追加してください。

---

### A) Noise σ Predictor

#### 1) `configs/noise_predictor.yml`（新規作成）

```yaml
data:
  data_dir: "./datasets/data/ffhq"   # ←あなたの環境に合わせて
  num_workers: 4

train:
  batch_size: 32
  learning_rate: 1.0e-4
  num_epochs: 50
  log_freq: 50
  save_freq: 5

sampling:
  batch_size: 1
```

#### 2) 学習実行

```bash
python train_noise_predictor.py --config noise_predictor.yml
```

---

### B) Degradation Type Predictor

#### 1) `configs/degradation_type_predictor.yml`（新規作成）

```yaml
data:
  data_dir: "./datasets/data/ffhq"
  num_workers: 4

train:
  batch_size: 32
  learning_rate: 1.0e-4
  num_epochs: 50
  log_freq: 50
  save_freq: 5

sampling:
  batch_size: 1
```

#### 2) 学習実行

```bash
python train_degradation_type_predictor.py --config degradation_type_predictor.yml
```

---

### C) Blur Predictor（length, angle）

#### 1) `configs/blur_predictor.yml`（新規作成）

```yaml
data:
  data_dir: "./datasets/data/ffhq"
  num_workers: 4

train:
  batch_size: 32
  learning_rate: 1.0e-4
  num_epochs: 50
  log_freq: 50
  save_freq: 5

sampling:
  batch_size: 1
```

#### 2) 学習実行

```bash
python train_blur.py --config blur_predictor.yml
```

---

## Metrics（PSNR / SSIM）

```bash
python calculate_psnr_ssim.py
```

> 補足：`calculate_psnr_ssim.py` が GT/出力のパスを環境依存で持っている場合があります。必要に応じて `gt_path` と `results_path` を自分の環境に合わせてください。

---

## Results（PPTの内容サマリ）

### Degradation Estimation

| Task | Metric | Result |
|---|---:|---:|
| Degradation type estimation | Accuracy ↑ | **100.00%** |
| Noise level estimation | RMSE ↓ | **0.50** |
| Motion blur length estimation | MSE ↓ | **0.008** |
| Motion blur angle estimation | MSE ↓ | **0.004** |

### Restoration（PSNR / SSIM）

以下は **w/o degradation info** と **提案手法（ours, conditional）** の比較例です。

| Degradation | PSNR (w/o) | SSIM (w/o) | PSNR (ours) | SSIM (ours) |
|---|---:|---:|---:|---:|
| Noise | 34.42 | 0.878 | **36.13** | **0.903** |
| Motion blur | 33.78 | 0.911 | **35.21** | **0.917** |
| Downsample | 33.12 | 0.892 | **34.16** | 0.890 |

---

## Notes（就活提出で効くポイント）

- README冒頭で「何をやって、どう良くしたか」が1分で分かる構成にしています
- 再現手順（config名・コマンド）を実ファイル名で記載しています
- `__pycache__` や `.env` 等は commit しない（`.gitignore` 推奨）

---

## License

TBD
