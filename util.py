# from https://github.com/EmilienDupont/coin

import torch
import numpy as np

def exp_golomb_bits(x):
    """
    Exponential-Golomb 코딩의 비트 수를 추정합니다
    x: 양자화된 정수 텐서
    """
    x = x.abs().long()
    # Signed mapping: 0 -> 0, 1 -> 1, -1 -> 2, 2 -> 3, -2 -> 4...
    x_mapped = torch.where(x > 0, 2 * x - 1, 2 * x.abs())
    # Bits = 2 * floor(log2(x+1)) + 1
    return (2 * torch.floor(torch.log2(x_mapped + 1)) + 1).sum().item()

def estimate_total_bits(model, q_step, seed_bits=16):
    """
    양자화된 파라미터와 시드 비트를 합산하여 전체 비트를 계산합니다
    """
    total_bits = 0
    for p in model.parameters():
        if p.requires_grad:
            # 양자화 수행: p_q = round(p / q)
            p_quantized = torch.round(p / q_step)
            total_bits += exp_golomb_bits(p_quantized)
    
    return total_bits + seed_bits

def get_quantized_model_state(model, q_step):
    """특정 양자화 스텝이 적용된 모델의 state_dict를 반환합니다."""
    new_state = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            new_state[name] = torch.round(param / q_step) * q_step
        else:
            new_state[name] = param.clone()
    return new_state

def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0: return float('inf')
    return 20. * torch.log10(torch.tensor(1.0)) - 10. * torch.log10(mse)

def clamp_and_quantize_image(img):
    """8-bit 반영"""
    return torch.round(torch.clamp(img, 0., 1.) * 255) / 255.