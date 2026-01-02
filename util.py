# from https://github.com/EmilienDupont/coin

import torch
import torch.nn.functional as F


def exp_golomb_bits(x):
    """Exponential-Golomb (k=0) 비트 길이 추정"""
    x_abs = torch.round(x).abs().long()
    # 부호 있는 정수를 부호 없는 정수로 매핑
    x_mapped = torch.where(x > 0, 2 * x_abs - 1, 2 * x_abs)
    # 비트 길이 계산: 2 * floor(log2(x_mapped + 1)) + 1
    bits = (2 * torch.floor(torch.log2(x_mapped.float() + 1.0)) + 1.0)
    return bits.sum().item()

def estimate_bpp(model, q_step, num_pixels, seed_bits=16):
    """양자화된 파라미터와 시드 비트를 포함한 BPP 계산"""
    total_bits = 0
    for p in model.parameters():
        if p.requires_grad:
            total_bits += exp_golomb_bits(p / q_step)
    return (total_bits + seed_bits) / num_pixels

def get_quantized_state(model, q_step):
    """
    모든 state_dict 키(가중치 + 버퍼)를 포함하여 반환합니다.
    가중치에만 양자화를 적용하고 버퍼는 그대로 유지합니다.
    """
    current_state = model.state_dict()
    new_state = {}
    
    # 모델의 모든 파라미터 이름을 가져옴
    param_names = {n for n, p in model.named_parameters()}
    
    for name, val in current_state.items():
        if name in param_names:
            # 학습 가능한 파라미터만 양자화 수행
            new_state[name] = torch.round(val / q_step) * q_step
        else:
            # z_m, fixed_input 등 버퍼는 양자화 없이 그대로 복사 
            new_state[name] = val.clone()
    return new_state

def psnr(img1, img2):
    """두 이미지 사이의 PSNR 계산"""
    mse = F.mse_loss(img1, img2)
    if mse == 0: return float('inf')
    return 20. * torch.log10(torch.tensor(1.0).to(img1.device)) - 10. * torch.log10(mse)

def clamp_image(img):
    """8-bit 이미지 표현 반영"""
    return torch.round(torch.clamp(img, 0., 1.) * 255) / 255.