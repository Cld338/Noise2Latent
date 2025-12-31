import os
import glob
import json
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
import copy
from PIL import Image

from model import NoiseToLatentModel
from .util import *

def train_one_image(image_path, config, epochs, target_bitwidth=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_name = os.path.basename(image_path)
    
    img_pil = Image.open(image_path).convert('RGB')
    target = T.ToTensor()(img_pil).unsqueeze(0).to(device)
    _, _, h, w = target.shape
    
    model = NoiseToLatentModel(h, w, config, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_psnr_val = -1.0
    best_model_state = None
    
    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        recon = model()
        
        loss = F.mse_loss(recon, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 100 == 0:
            model.eval()
            with torch.no_grad():
                cur_recon = model()
                clamped_psnr = psnr(target, clamp_image(cur_recon))
                
                if clamped_psnr > best_psnr_val:
                    best_psnr_val = clamped_psnr
                    best_model_state = copy.deepcopy(model.state_dict())
            model.train()

        if epoch % 1000 == 0:
            print(f"[{image_name}] Ep {epoch} | Loss: {loss.item():.6f} | Best PSNR: {best_psnr_val:.2f}dB")
            
    model.load_state_dict(best_model_state)
    
    if target_bitwidth == 16:
        model = model.half()
        target = target.half()
    
    model.eval()
    with torch.no_grad():
        final_output = model()
        final_recon = clamp_image(final_output)
        final_psnr = psnr(target.float(), final_recon.float())
        final_bpp = bpp(target, model, target_bitwidth)
            
    return final_bpp, final_psnr

def save_results_to_json(results_dict, save_path='rd_results.json'):
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)

def plot_final_rd_curve(results_dict, save_path='final_rd_curve.png'):
    plt.figure(figsize=(10, 7))
    plot_data = []
    for setting_name, values in results_dict.items():
        if not values: continue
        avg_bpp = sum([v[0] for v in values]) / len(values)
        avg_psnr = sum([v[1] for v in values]) / len(values)
        plot_data.append((setting_name, avg_bpp, avg_psnr))
    
    plot_data.sort(key=lambda x: x[1]) # BPP 순 정렬
    
    if plot_data:
        bpps = [d[1] for d in plot_data]
        psnrs = [d[2] for d in plot_data]
        labels = [d[0] for d in plot_data]
        plt.plot(bpps, psnrs, marker='o', markersize=8, linewidth=2, label='Proposed (Kodak Mean)')
        for i, label in enumerate(labels):
            plt.annotate(label, (bpps[i], psnrs[i]), xytext=(5, -5), textcoords='offset points')

    plt.title('Rate-Distortion Performance (16-bit Weights & 8-bit Image)')
    plt.xlabel('Rate (bpp)')
    plt.ylabel('Distortion (PSNR dB)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig(save_path)
    plt.show()

def main():
    data_dir = './data/kodak_dataset'
    image_paths = sorted(glob.glob(os.path.join(data_dir, '*.png')))
    
    if not image_paths:
        print(f"데이터셋을 찾을 수 없습니다: {data_dir}")
        return

    experiment_configs = [
        {'name': 'S0 (Very Small)', 'scales': 4, 'nch': 8, 'cch': 8, 'pe_dims': 8, 'M': 2, 'N': 2, 'seed': 42},
        {'name': 'S1 (Small)', 'scales': 4, 'nch': 12, 'cch': 12, 'pe_dims': 10, 'M': 2, 'N': 2, 'seed': 42},
        {'name': 'S2 (Medium)', 'scales': 4, 'nch': 12, 'cch': 16, 'pe_dims': 12, 'M': 3, 'N': 3, 'seed': 42},
        {'name': 'S3 (Large)', 'scales': 4, 'nch': 16, 'cch': 24, 'pe_dims': 16, 'M': 4, 'N': 4, 'seed': 42},
    ]
    
    epochs = 15000 
    results_dict = {cfg['name']: [] for cfg in experiment_configs}

    for cfg in experiment_configs:
        print(f"\n--- 현재 세팅: {cfg['name']} ---")
        for img_path in image_paths:
            res_bpp, res_psnr = train_one_image(img_path, cfg, epochs, target_bitwidth=16)
            results_dict[cfg['name']].append((res_bpp, res_psnr))
            
            save_results_to_json(results_dict)
            plot_final_rd_curve(results_dict)

if __name__ == '__main__':
    main()