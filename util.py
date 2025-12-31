import torch
import numpy as np
from typing import Dict

DTYPE_BIT_SIZE: Dict[torch.dtype, int] = {
    torch.float32: 32,
    torch.float: 32,
    torch.float16: 16,
    torch.half: 16,
    torch.bfloat16: 16,
    torch.uint8: 8,
    torch.int8: 8,
}

def model_size_in_bits(model, target_bitwidth=16):
    """
    모델의 학습 가능한 파라미터 비트 수를 계산합니다.
    복원기에서 시드(Seed)로 재생성 가능한 buffer(z_m, grid)는 계산에서 제외합니다.
    """
    total_params = sum(p.nelement() for p in model.parameters() if p.requires_grad)
    return total_params * target_bitwidth

def bpp(image, model, target_bitwidth=16):
    """이미지 해상도 대비 BPP(Bits Per Pixel)를 계산합니다."""
    if len(image.shape) == 4:
        _, _, h, w = image.shape
    else:
        _, h, w = image.shape
    num_pixels = h * w
    return model_size_in_bits(model, target_bitwidth) / num_pixels

def psnr(img1, img2):
    """두 이미지 사이의 PSNR을 계산합니다."""
    mse = (img1 - img2).detach().pow(2).mean()
    if mse == 0:
        return float('inf')
    return 20. * np.log10(1.) - 10. * mse.log10().to('cpu').item()

def clamp_image(img):
    """이미지를 [0, 1] 범위로 클램핑하고 8비트(256단계) 양자화 효과를 반영합니다."""
    img_ = torch.clamp(img, 0., 1.)
    return torch.round(img_ * 255) / 255.
