import math
import torch
import torch.nn as nn


def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


# ★★★ 追加: DegradationEncoder ★★★
class DegradationEncoder(nn.Module):
    def __init__(self, type_dim=3, param_dim=4, embedding_dim=512, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        
        # ワンホットベクトル用のエンコーダー
        self.type_encoder = nn.Sequential(
            nn.Linear(type_dim, embedding_dim // 2),
            nn.SiLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        # 4次元パラメータ用のエンコーダー
        self.param_encoder = nn.Sequential(
            nn.Linear(param_dim, embedding_dim // 2),
            nn.SiLU(),
            nn.Linear(embedding_dim // 2, embedding_dim // 2)
        )
        
        # 統合してトークン列に変換
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * num_tokens),
            nn.SiLU(),
        )
        
        self.embedding_dim = embedding_dim
    
    def forward(self, degradation_type, degradation_params):
        """
        Args:
            degradation_type: (B, 3) ワンホットベクトル
            degradation_params: (B, 4) パラメータベクトル
        Returns:
            (B, num_tokens, embedding_dim) トークン列
        """
        type_emb = self.type_encoder(degradation_type)  # (B, 256)
        param_emb = self.param_encoder(degradation_params)  # (B, 256)
        
        combined = torch.cat([type_emb, param_emb], dim=-1)  # (B, 512)
        
        output = self.fusion(combined)  # (B, 4096)
        output = output.view(-1, self.num_tokens, self.embedding_dim)  # (B, 8, 512)
        
        return output

# ★★★ 追加: CrossAttention ★★★
class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim)
        self.to_k = nn.Linear(context_dim, query_dim)
        self.to_v = nn.Linear(context_dim, query_dim)
        self.to_out = nn.Linear(query_dim, query_dim)
        
    def forward(self, x, context):
        B, C, H, W = x.shape
        x_flat = x.reshape(B, C, H*W).permute(0, 2, 1)  # (B, H*W, C)
        
        q = self.to_q(x_flat)
        k = self.to_k(context)
        v = self.to_v(context)
        
        q = q.reshape(B, H*W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, H*W, C)
        out = self.to_out(out)
        out = out.permute(0, 2, 1).reshape(B, C, H, W)
        
        return out


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w).permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        w_ = torch.bmm(q, k)
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)
        h_ = self.proj_out(h_)
        return x+h_


class DiffusionUNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ★★★ 最初に定義 ★★★
        self.use_cross_attention = getattr(config.model, 'use_cross_attention', False)
        
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels * 2 if config.data.conditional else config.model.in_channels
        resolution = config.data.image_size
        resamp_with_conv = config.model.resamp_with_conv

        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # ★★★ DegradationEncoder ★★★
        if self.use_cross_attention:
            context_dim = getattr(config.model, 'context_dim', 512)
            num_heads = getattr(config.model, 'num_heads', 4)
            self.degradation_encoder = DegradationEncoder(
                type_dim=3,      # ワンホットベクトル
                param_dim=4,     # 4次元パラメータ
                embedding_dim=context_dim,
                num_tokens=8
            )

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch, self.temb_ch),
            torch.nn.Linear(self.temb_ch, self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        self.down_cross_attn = nn.ModuleList()
        block_in = None
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            cross_attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out,
                                        temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
                # ★★★ CrossAttention追加 ★★★
                if self.use_cross_attention:
                    cross_attn.append(CrossAttention(block_in, context_dim, num_heads))
                    
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)
            self.down_cross_attn.append(cross_attn)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        if self.use_cross_attention:
            self.mid.cross_attn = CrossAttention(block_in, context_dim, num_heads)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in,
                                       temb_channels=self.temb_ch, dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        self.up_cross_attn = nn.ModuleList()
        
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            cross_attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in, out_channels=block_out,
                                        temb_channels=self.temb_ch, dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
                # ★★★ CrossAttention追加 ★★★
                if self.use_cross_attention:
                    cross_attn.append(CrossAttention(block_in, context_dim, num_heads))
                    
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)
            self.up_cross_attn.insert(0, cross_attn)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, t, degradation_type=None, degradation_params=None):
        assert x.shape[2] == x.shape[3] == self.resolution

        # ★★★ 劣化情報からコンテキスト生成 ★★★
        context = None
        if self.use_cross_attention and degradation_type is not None and degradation_params is not None:
            context = self.degradation_encoder(degradation_type, degradation_params)

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                # ★★★ CrossAttention適用 ★★★
                if self.use_cross_attention and context is not None:
                    h = h + self.down_cross_attn[i_level][i_block](h, context)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        if self.use_cross_attention and context is not None:
            h = h + self.mid.cross_attn(h, context)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
                # ★★★ CrossAttention適用 ★★★
                if self.use_cross_attention and context is not None:
                    h = h + self.up_cross_attn[i_level][i_block](h, context)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h