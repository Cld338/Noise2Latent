import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import copy
import json
import os

from model_v2 import NoiseToLatentModel
from util import *


def train_and_search(image_path, config, epochs=1000, lmbda=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_pil = Image.open(image_path).convert('RGB')
    target = T.ToTensor()(img_pil).unsqueeze(0).to(device)
    _, _, h, w = target.shape
    
    model = NoiseToLatentModel(h, w, config).to(device)
    
    # Adam lr 8e-3
    # BPP 감소를 위해 Weight Decay 추가
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3) # weight_decay=1e-5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"Starting training for {epochs} epochs...")
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        recon = model()
        loss = F.mse_loss(recon, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (ep + 1) % 100 == 0:
            print(f"Epoch {ep+1}/{epochs} | Loss: {loss.item():.6f}")

    # Mesh Search: 최적의 양자화 스텝 q 탐색
    # D + lambda * R 비용이 최소가 되는 지점 선택
    candidate_qs = [2**-i for i in range(6, 15)] 
    best_rd_cost = float('inf')
    best_stats = {"bpp": 0, "psnr": 0}
    
    model.eval()
    with torch.no_grad():
        # 원본 상태 저장
        trained_state = copy.deepcopy(model.state_dict())
        for q in candidate_qs:
            # 버퍼가 포함된 전체 state_dict 로드
            model.load_state_dict(get_quantized_state(model, q))
            recon = clamp_image(model())
            
            cur_psnr = psnr(target, recon).item()
            cur_bpp = estimate_bpp(model, q, h * w)
            dist = F.mse_loss(recon, target).item()
            
            # Rate-Distortion 최적화
            rd_cost = dist + lmbda * cur_bpp
            
            if rd_cost < best_rd_cost:
                best_rd_cost = rd_cost
                best_stats = {"bpp": cur_bpp, "psnr": cur_psnr}
        # model.load_state_dict(trained_state) 

    print(f"\n[Final Results]")
    print(f"BPP: {best_stats['bpp']:.4f} | PSNR: {best_stats['psnr']:.2f}dB")
    return best_stats


def run_experiment(dataset_path, config, epochs=2000, lmbda=0.005, save_json="results.json"):
    """
    지정된 폴더 내의 모든 이미지에 대해 학습을 수행하고 결과를 JSON 파일로 저장합니다.
    

    """
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(valid_extensions)])
    
    if not image_files:
        print(f"No images found in {dataset_path}")
        return

    experiment_data = {
        "config": config,
        "hyperparameters": {
            "epochs": epochs,
            "lmbda": lmbda
        },
        "results": [],
        "summary": {}
    }

    print(f"Starting experiment on {len(image_files)} images...")

    total_bpp = 0
    total_psnr = 0
    count = 0

    for img_name in image_files:
        img_path = os.path.join(dataset_path, img_name)
        print(f"\n>>> Processing: {img_name}")
        
        try:
            stats = train_and_search(img_path, config, epochs=epochs, lmbda=lmbda)
            
            result_entry = {
                "image_name": img_name,
                "bpp": round(stats['bpp'], 6),
                "psnr": round(stats['psnr'], 4)
            }
            experiment_data["results"].append(result_entry)
            
            total_bpp += stats['bpp']
            total_psnr += stats['psnr']
            count += 1

            avg_bpp = total_bpp / count
            avg_psnr = total_psnr / count
            experiment_data["summary"] = {
                "avg_bpp": round(avg_bpp, 6),
                "avg_psnr": round(avg_psnr, 4),
                "count": count
            }

            with open(save_json, 'w', encoding='utf-8') as f:
                json.dump(experiment_data, f, indent=4)
                
        except Exception as e:
            print(f"Error processing {img_name}: {e}")

    print(f"\n[Experiment Finished] Results saved to {save_json}")
    return experiment_data


if __name__ == "__main__":
    config_s0 = {'scales': 4, 'nch': 12, 'cch': 8, 'pe_dims': 8, 'M': 3, 'N': 3, 'seed': 42}
    # train_and_search('C:/workspace/Noise2Latent/data/kodak_dataset/kodim01.png', config_s0, epochs=20000)
    run_experiment("./data/kodak_dataset/", config_s0, 20000, 0.005)