import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import copy
from model import NoiseToLatentModel
from util import *

def train_and_search(image_path, config, epochs=1000, lmbda=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_pil = Image.open(image_path).convert('RGB')
    target = T.ToTensor()(img_pil).unsqueeze(0).to(device)
    _, _, h, w = target.shape
    
    model = NoiseToLatentModel(h, w, config).to(device)
    
    # Adam lr 8e-3
    # BPP 감소를 위해 Weight Decay 추가
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3, weight_decay=1e-5) 
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


if __name__ == "__main__":
    config_s0 = {'scales': 4, 'nch': 12, 'cch': 8, 'pe_dims': 8, 'M': 3, 'N': 3, 'seed': 42}
    train_and_search('C:/workspace/Noise2Latent/data/kodak_dataset/kodim01.png', config_s0, epochs=20000)