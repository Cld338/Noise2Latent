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

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        # Receptive Field 확장을 위해 커널 사이즈 7 적용
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # 확장 비율 4배
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # 학습 안정성을 위한 LayerScale
        self.gamma = nn.Parameter(1e-6 * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return input + x

class NoiseToLatentModel(nn.Module):
    def __init__(self, h, w, config):
        super().__init__()
        self.latent_channels = config['scales'] * config['nch']
        self.M = config['M'] # GPP 블록 수
        self.N = config['N'] # Synthesis 블록 수
        
        # Noise Pyramid (고정 시드)
        z_m = self._generate_noise_pyramid(h, w, config)
        self.register_buffer("z_m", z_m)
        
        pe_layer = PositionalEncoding(config['pe_dims'], h, w)
        self.register_buffer("fixed_input", torch.cat([z_m, pe_layer()], dim=1))
        
        # GPP: Gaussian Parameter Predictor
        gpp_in = self.latent_channels + config['pe_dims']
        self.gpp_initial = nn.Conv2d(gpp_in, config['cch'], 1)
        # Sequential 대신 ModuleList 사용하여 중간 특징 추출 가능하게 함
        self.gpp_blocks = nn.ModuleList([ConvNeXtBlock(config['cch']) for _ in range(self.M)])
        
        self.mu_head = nn.Conv2d(config['cch'], self.latent_channels, 1)
        self.sigma_head = nn.Conv2d(config['cch'], self.latent_channels, 1)
        
        # Synthesis Network
        self.syn_initial = nn.Conv2d(self.latent_channels, config['cch'], 1)
        self.syn_blocks = nn.ModuleList([ConvNeXtBlock(config['cch']) for _ in range(self.N)])
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
        # 1. GPP Pass
        feat = self.gpp_initial(self.fixed_input)
        gpp_skips = []
        for block in self.gpp_blocks:
            feat = block(feat)
            gpp_skips.append(feat) # 각 블록의 출력을 저장
            
        # 2. Gaussian Latent Sampling
        mu = self.mu_head(feat)
        sigma = F.softplus(self.sigma_head(feat)) + 1e-6
        y_pred = mu + sigma * self.z_m
        
        # 3. Synthesis Pass (with Residual Connections)
        out = self.syn_initial(y_pred)
        
        # M과 N이 다를 수 있으므로 대응하는 skip만 사용 (뒤에서부터 연결)
        for i, block in enumerate(self.syn_blocks):
            # GPP의 특징을 Synthesis 특징에 더해줌 (Residual)
            if i < len(gpp_skips):
                # 역순 혹은 순차적으로 연결 가능 (여기서는 순차 연결)
                out = out + gpp_skips[i] 
            out = block(out)
            
        return torch.sigmoid(self.syn_final(out))