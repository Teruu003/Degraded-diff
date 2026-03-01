import torch
import shutil
import os
import torchvision.utils as tvu


def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    tvu.save_image(img, file_directory)


def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    """
    チェックポイントを読み込む
    
    ★★★ 修正: PyTorch 2.6以降に対応 ★★★
    weights_only=False を追加して、argparse.Namespaceなどの
    カスタムオブジェクトを含むチェックポイントを読み込めるようにする
    
    注意: 信頼できるソースからのチェックポイントにのみ使用すること
    """
    if device is None:
        return torch.load(path, weights_only=False)
    else:
        return torch.load(path, map_location=device, weights_only=False)
