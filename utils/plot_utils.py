# utils/plot_utils.py
import matplotlib.pyplot as plt
import os

def plot_loss(steps, losses, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label=title)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close() # メモリ解放

def plot_multiple_losses(histories, title, xlabel, ylabel, save_path):
    plt.figure(figsize=(12, 7))
    for label, (steps, losses) in histories.items():
        if len(steps) > 0 and len(losses) > 0: # データが存在する場合のみプロット
            plt.plot(steps, losses, label=label)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close() # メモリ解放