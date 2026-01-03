import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, out_dims, h, w):
        super().__init__()
        self.out_dims = out_dims
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0)
        self.register_buffer("grid", grid)

    def forward(self):
        pe_list = [self.grid]
        num_frequencies = math.ceil((self.out_dims - 2) / 4)
        for i in range(num_frequencies):
            for func in [torch.sin, torch.cos]:
                pe_list.append(func(self.grid * (2**i) * math.pi))
        pe = torch.cat(pe_list, dim=1)
        return pe[:, :self.out_dims, :, :]

class LiteConvNeXtBlock(nn.Module):
    """BPP 절감을 위해 Depthwise Separable 구조와 낮은 확장 비율을 적용한 블록"""
    def __init__(self, dim):
        super().__init__()
        # Depthwise Conv (7x7로 수용 영역은 유지)
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Expansion ratio를 1.5배로 제한하여 파라미터 절감
        hidden_dim = int(1.5 * dim)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return input + x

class GlobalChannelModulator(nn.Module):
    """채널별 글로벌 스케일링을 수행하는 극소량의 파라미터 필터 (Fourier Domain)"""
    def __init__(self, channels):
        super().__init__()
        # 채널당 1개의 복소수 가중치만 사용 (BPP 영향 최소화)
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1, 2))

    def forward(self, x):
        h, w = x.shape[-2:]
        x_fft = torch.fft.rfft2(x, norm='ortho')
        weight = torch.view_as_complex(self.weight)
        x_fft = x_fft * weight # 채널별 글로벌 주파수 조정
        return torch.fft.irfft2(x_fft, s=(h, w), norm='ortho')

class NoiseToLatentModel(nn.Module):
    def __init__(self, h, w, config):
        super().__init__()
        self.h, self.w = h, w
        self.latent_channels = config['scales'] * config['nch']
        
        # Noise Pyramid (고정 시드)
        z_m = self._generate_noise_pyramid(h, w, config)
        self.register_buffer("z_m", z_m)
        
        pe_layer = PositionalEncoding(config['pe_dims'], h, w)
        self.register_buffer("fixed_input", torch.cat([z_m, pe_layer()], dim=1))
        
        # 1. GPP: Gaussian Parameter Predictor (경량화 버전)
        gpp_in = self.latent_channels + config['pe_dims']
        self.gpp_initial = nn.Conv2d(gpp_in, config['cch'], 1)
        # 모듈 수를 유지하되 내부 파라미터를 줄인 LiteConvNeXtBlock 사용
        self.gpp_blocks = nn.Sequential(*[LiteConvNeXtBlock(config['cch']) for _ in range(config['M'])])
        
        # 2. 글로벌 변조 레이어 (이미지 전역 특성 반영)
        self.global_modulator = GlobalChannelModulator(config['cch'])
        
        # 3. Heads
        self.mu_head = nn.Conv2d(config['cch'], self.latent_channels, 1)
        self.sigma_head = nn.Conv2d(config['cch'], self.latent_channels, 1)
        
        # 4. Synthesis Network
        self.syn_initial = nn.Conv2d(self.latent_channels, config['cch'], 1)
        self.syn_blocks = nn.Sequential(*[LiteConvNeXtBlock(config['cch']) for _ in range(config['N'])])
        self.syn_final = nn.Conv2d(config['cch'], 3, 1)

    def _generate_noise_pyramid(self, h, w, config):
        generator = torch.Generator().manual_seed(config['seed'])
        noises = []
        for i in range(config['scales']):
            s_h, s_w = h // (2**i), w // (2**i)
            z_i = torch.randn(1, config['nch'], s_h, s_w, generator=generator)
            z_i_up = F.interpolate(z_i, size=(h, w), mode='bilinear', align_corners=False)
            noises.append(z_i_up)
        return torch.cat(noises, dim=1)

    def forward(self):
        # GPP 과정
        feat = self.gpp_initial(self.fixed_input)
        feat = self.gpp_blocks(feat)
        
        # 글로벌 채널 변조 적용 (BPP 추가 없이 성능 향상)
        feat = self.global_modulator(feat)
        
        mu = self.mu_head(feat)
        sigma = F.softplus(self.sigma_head(feat)) + 1e-6
        
        # Reparameterization Trick [cite: 149]
        y_pred = mu + sigma * self.z_m
        
        # Synthesis 과정
        out = self.syn_initial(y_pred)
        out = self.syn_blocks(out)
        return torch.sigmoid(self.syn_final(out))