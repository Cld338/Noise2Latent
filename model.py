import torch 
import math
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, out_dims, h, w):
        super().__init__()
        self.out_dims = out_dims
        grid_y, grid_x = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing='ij')
        grid = torch.stack([grid_y, grid_x], dim=0).unsqueeze(0) # [1, 2, H, W]
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
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

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

class NoiseToLatentModel(nn.Module):
    def __init__(self, h, w, config):
        super().__init__()
        self.scales = config['scales']
        self.nch = config['nch'] # 12
        self.latent_channels = self.scales * self.nch
        
        # 1. noise_pyramid 생성
        z_m = self._generate_noise_pyramid(h, w, config)
        self.register_buffer("z_m", z_m)
        
        # 2. Positional Encoding
        pe_layer = PositionalEncoding(config['pe_dims'], h, w)
        self.register_buffer("pe", pe_layer())
        
        # 3. GPP (M blocks)
        gpp_in = self.latent_channels + config['pe_dims']
        self.gpp_initial = nn.Conv2d(gpp_in, config['cch'], 3, padding=1)
        self.gpp_blocks = nn.Sequential(*[ConvNeXtBlock(config['cch']) for _ in range(config['M'])])
        self.mu_head = nn.Conv2d(config['cch'], self.latent_channels, 1)
        self.sigma_head = nn.Conv2d(config['cch'], self.latent_channels, 1)
        
        # 4. Synthesis Network (N blocks) [cite: 103, 156]
        self.syn_initial = nn.Conv2d(self.latent_channels, config['cch'], 3, padding=1)
        self.syn_blocks = nn.Sequential(*[ConvNeXtBlock(config['cch']) for _ in range(config['N'])])
        self.syn_final = nn.Conv2d(config['cch'], 3, 3, padding=1)

    def _generate_noise_pyramid(self, h, w, config):
        generator = torch.Generator().manual_seed(config['seed'])
        noises = []
        for i in range(self.scales):
            s_h, s_w = h // (2**i), w // (2**i)
            z_i = torch.randn(1, self.nch, s_h, s_w, generator=generator)
            z_i_up = F.interpolate(z_i, size=(h, w), mode='bilinear', align_corners=False)
            noises.append(z_i_up)
        return torch.cat(noises, dim=1) # [1, 4*12, H, W]

    def forward(self):
        # Noise + PE concat
        gpp_input = torch.cat([self.z_m, self.pe], dim=1)
        feat = self.gpp_initial(gpp_input)
        feat = self.gpp_blocks(feat)
        
        mu = self.mu_head(feat)
        sigma = F.softplus(self.sigma_head(feat)) + 1e-6
        
        # Reparameterization
        y_pred = mu + sigma * self.z_m
        
        out = self.syn_initial(y_pred)
        out = self.syn_blocks(out)
        return torch.sigmoid(self.syn_final(out))