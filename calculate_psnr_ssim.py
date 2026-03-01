import os
import cv2
from utils.metrics import calculate_psnr, calculate_ssim # これらの関数が正しくインポートされている前提

# 設定
# GT画像が保存されている元のフォルダ
gt_path = "C:/Users/yuzu0/Documents/zemi_2025spring/datasets/data/ffhq/test/downsample_gt"
# 復元画像が保存されているフォルダ（models/restoration.py で指定したパス）
results_path = "C:/Users/yuzu0/Documents/zemi_2025spring/WeatherDiffusion-main/results/images/FFHQ/downsample_restored"

cumulative_psnr, cumulative_ssim = 0, 0
count = 0

# results_path フォルダからファイル名をリストアップ
# ここで ['04000'].png のようなファイル名がリストに含まれると想定
restored_filenames_raw = sorted(os.listdir(results_path))

for raw_filename in restored_filenames_raw:
    # 1. 余計な文字を削除して、純粋なファイル名を取得
    # 例: "['04000'].png" -> "04000.png"
    processed_filename = raw_filename.replace("['", "").replace("']", "")
    
    # 2. 拡張子を除去してベースファイル名を取得 (例: "04000.png" -> "04000")
    base_filename_without_ext = os.path.splitext(processed_filename)[0]

    # 3. GT画像のファイル名を取得 (GTは通常、拡張子を含まないベースファイル名 + '.png' など)
    # FFHQのファイル名が '000000.png' のような形式であると仮定
    gt_filename = f"{base_filename_without_ext}.png" 
    
    # フルパスを構築
    gt_full_path = os.path.join(gt_path, gt_filename)
    restored_full_path = os.path.join(results_path, raw_filename) # 復元された画像はそのままのファイル名で読み込む

    # ファイルの存在確認
    if not os.path.exists(restored_full_path):
        print(f"Warning: Restored image not found: {restored_full_path}. Skipping.")
        continue
    if not os.path.exists(gt_full_path):
        print(f"Warning: Ground Truth image not found: {gt_full_path}. Skipping.")
        continue

    print('Processing image: %s (Restored) vs %s (GT)' % (raw_filename, gt_filename))
    
    res = cv2.imread(restored_full_path, cv2.IMREAD_COLOR)
    gt = cv2.imread(gt_full_path, cv2.IMREAD_COLOR)

    if res is None or gt is None:
        print(f"Error reading images {restored_full_path} or {gt_full_path}. Skipping.")
        continue
    
    # PSNRとSSIMの計算
    cur_psnr = calculate_psnr(res, gt, test_y_channel=True)
    cur_ssim = calculate_ssim(res, gt, test_y_channel=True)
    print('PSNR is %.4f and SSIM is %.4f' % (cur_psnr, cur_ssim))
    cumulative_psnr += cur_psnr
    cumulative_ssim += cur_ssim
    count += 1

# 結果の出力
if count > 0:
    print('Testing set, PSNR is %.4f and SSIM is %.4f' % (cumulative_psnr / count, cumulative_ssim / count))
else:
    print("No images processed for metrics calculation.")
print(f"Results folder: {results_path}")