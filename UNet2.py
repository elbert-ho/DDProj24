import math
from abc import abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_nd(dims, in_channels, out_channels, kernel_size, padding=0, stride=1):
    if dims == 1:
        return nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, stride=stride)
    else:
        raise ValueError("Only 1D is supported in this adaptation")

def linear(in_features, out_features):
    return nn.Linear(in_features, out_features)

def normalization(channels, swish=1.0):
    return nn.BatchNorm1d(channels)

def zero_module(module):
    nn.init.constant_(module.weight, 0)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
    return module

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, encoder_out=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    def __init__(self, channels, use_conv, dims=1):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, channels, use_conv, dims=1):
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        if use_conv:
            self.conv = conv_nd(dims, channels, channels, 3, padding=1, stride=2)
        else:
            self.conv = nn.AvgPool1d(2)

    def forward(self, x):
        return self.conv(x)

class ResBlock(TimestepBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False,
                 use_scale_shift_norm=False, dims=1, use_checkpoint=False, up=False, down=False):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1)
        )

        self.updown = up or down
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(emb_channels, 2 * self.out_channels if use_scale_shift_norm else self.out_channels)
        )

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1))
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 3, padding=1)
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    def __init__(self, channels, num_heads=1, use_checkpoint=False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttention(self.num_heads)
        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        b, c, length = x.shape
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(b, 3 * self.num_heads, c // self.num_heads, length).permute(0, 2, 3, 1)
        h = self.attention(qkv)
        h = self.proj_out(h.permute(0, 3, 1, 2).reshape(b, c, length))
        return x + h

class QKVAttention(nn.Module):
    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        b, h, t, c = qkv.shape
        q, k, v = qkv.chunk(3, dim=-1)
        q = q * (c ** -0.5)
        weight = torch.einsum('bhtd,bhfd->bhtf', q, k).softmax(dim=-1)
        a = torch.einsum('bhtf,bhfd->bhtd', weight, v)
        return a

class UNetModel(nn.Module):
    def __init__(self, in_channels, model_channels, out_channels, num_res_blocks, attention_resolutions,
                 dropout=0, channel_mult=(32768, 4096, 512, 64), conv_resample=True, dims=1, num_classes=None, use_checkpoint=False):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim)
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        ch = input_ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            TimestepEmbedSequential(conv_nd(dims, in_channels, ch, 3, padding=1))
        ])
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch, time_embed_dim, dropout, out_channels=int(mult * model_channels),
                        dims=dims, use_checkpoint=use_checkpoint
                    )
                ]
                ch = int(mult * model_channels)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint))
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch,
                            dims=dims, use_checkpoint=use_checkpoint, down=True
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint
            ),
            AttentionBlock(
                ch, use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch, time_embed_dim, dropout, dims=dims, use_checkpoint=use_checkpoint
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich, time_embed_dim, dropout, out_channels=int(model_channels * mult),
                        dims=dims, use_checkpoint=use_checkpoint
                    )
                ]
                ch = int(model_channels * mult)
                if ds in attention_resolutions:
                    layers.append(AttentionBlock(ch, use_checkpoint=use_checkpoint))
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch, time_embed_dim, dropout, out_channels=out_ch,
                            dims=dims, use_checkpoint=use_checkpoint, up=True
                        )
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, input_ch, out_channels, 3, padding=1))
        )

    def forward(self, x, timesteps, y=None):
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        if self.num_classes is not None:
            assert y is not None
            emb = emb + self.label_emb(y)

        hs = []
        h = x
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        return self.out(h)

def timestep_embedding(timesteps, dim):
    half = dim // 2
    emb = matorch.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb