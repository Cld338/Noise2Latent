import torch
import torch.nn.functional as F
from torchvision import transforms as T
from PIL import Image
import copy

from model import NoiseToLatentModel
from util import *

def train_and_search(image_path, config, epochs=50000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 이미지 로드
    img_pil = Image.open(image_path).convert('RGB')
    target = T.ToTensor()(img_pil).unsqueeze(0).to(device)
    _, _, h, w = target.shape
    
    model = NoiseToLatentModel(h, w, config).to(device)
    
    # 논문 설정: lr 8e-3, Adam, Cosine Annealing
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 1. 오버피팅 학습 (MSE Loss)
    model.train()
    for ep in range(epochs):
        optimizer.zero_grad()
        recon = model()
        loss = F.mse_loss(recon, target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (ep + 1) % 100 == 0:
            print(f"Epoch {ep+1} Loss: {loss.item():.6f}")

    # 2. Mesh Search (최적 양자화 스텝 찾기)
    # 후보 스텝: 2^-k 형태
    candidate_steps = [2**-i for i in range(4, 14)] 
    best_rd_cost = float('inf')
    best_q = None
    best_psnr_val = 0
    best_bpp_val = 0
    
    # 논문은 RD trade-off (Loss = D + lambda * R)를 위해 lambda가 필요하지만, 
    # 여기서는 고정된 설정에서 최적의 PSNR을 내는 Q를 찾는 방식
    # 논문에서는 특정 lambda에 대해 탐색함.
    
    model.eval()
    with torch.no_grad():
        original_state = copy.deepcopy(model.state_dict())
        for q in candidate_steps:
            # 양자화 적용
            model.load_state_dict(get_quantized_model_state(model, q))
            recon = clamp_and_quantize_image(model())
            
            cur_psnr = psnr(target, recon).item()
            cur_bits = estimate_total_bits(model, q)
            cur_bpp = cur_bits / (h * w)
            
            # 임의의 lambda (예: 0.01) 설정 시 RD cost 계산
            rd_cost = F.mse_loss(recon, target).item() + 0.01 * cur_bpp
            
            if rd_cost < best_rd_cost:
                best_rd_cost = rd_cost
                best_q = q
                best_psnr_val = cur_psnr
                best_bpp_val = cur_bpp
        
        model.load_state_dict(get_quantized_model_state(model, best_q))
        
    print(f"Best Q-Step: {best_q} | PSNR: {best_psnr_val:.2f}dB | BPP: {best_bpp_val:.4f}")
    return best_bpp_val, best_psnr_val

if __name__ == "__main__":
    # 논문 Table 1 - Setting 0 (Very Small) 재현 
    config_s0 = {
        'scales': 4,
        'nch': 12,
        'cch': 8, 
        'pe_dims': 8,
        'M': 3,
        'N': 3,
        'seed': 42
    }
    
    # 실험 실행
    train_and_search('data/kodak/kodim01.png', config_s0)