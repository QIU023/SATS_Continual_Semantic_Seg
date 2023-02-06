from math import sqrt
from functools import partial
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

from modules.segformer_offical.mix_transformer import *

# helpers

def exists(val):
    return val is not None

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth

# classes

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride = 1, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride, bias = bias),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1, bias = bias)
        )
    def forward(self, x):
        return self.net(x)

class LayerNorm(nn.Module):
    def __init__(self, dim, eps = 1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        std = torch.var(x, dim = 1, unbiased = False, keepdim = True).sqrt()
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) / (std + self.eps) * self.g + self.b

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x))
    
class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class EfficientSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        heads,
        reduction_ratio
    ):
        super().__init__()
        self.scale = (dim // heads) ** -0.5
        self.heads = heads

        self.to_q = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_kv = nn.Conv2d(dim, dim * 2, reduction_ratio, stride = reduction_ratio, bias = False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias = False)

    def forward(self, x):
        h, w = x.shape[-2:]
        heads = self.heads

        q, k, v = (self.to_q(x), *self.to_kv(x).chunk(2, dim = 1))
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> (b h) (x y) c', h = heads), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) (x y) c -> b (h c) x y', h = heads, x = h, y = w)
        return self.to_out(out)

class MixFeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor
    ):
        super().__init__()
        hidden_dim = dim * expansion_factor
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            DsConv2d(hidden_dim, hidden_dim, 3, padding = 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, dim, 1)
        )

    def forward(self, x):
        return self.net(x)

class MiT(nn.Module):
    def __init__(
        self,
        *,
        channels,
        dims,
        heads,
        ff_expansion,
        reduction_ratio,
        num_layers
    ):
        super().__init__()
        stage_kernel_stride_pad = ((7, 4, 3), (3, 2, 1), (3, 2, 1), (3, 2, 1))

        dims = (channels, *dims)
        dim_pairs = list(zip(dims[:-1], dims[1:]))
        
#         self.patch_embed = nn.ModuleList([
#             nn.Sequential(PatchEmbed(img_size=512), )
#         ])

        self.stages = nn.ModuleList([])

        for (dim_in, dim_out), (kernel, stride, padding), num_layers, ff_expansion, heads, reduction_ratio in zip(dim_pairs, stage_kernel_stride_pad, num_layers, ff_expansion, heads, reduction_ratio):
            get_overlap_patches = nn.Unfold(kernel, stride = stride, padding = padding)
            overlap_patch_embed = nn.Conv2d(dim_in * kernel ** 2, dim_out, 1)

            layers = nn.ModuleList([])

            for _ in range(num_layers):
                layers.append(nn.ModuleList([
                    PreNorm(dim_out, EfficientSelfAttention(dim = dim_out, heads = heads, reduction_ratio = reduction_ratio)),
                    PreNorm(dim_out, MixFeedForward(dim = dim_out, expansion_factor = ff_expansion)),
                ]))

            self.stages.append(nn.ModuleList([
                get_overlap_patches,
                overlap_patch_embed,
                layers
            ]))

    def forward(
        self,
        x,
        return_layer_outputs = False
    ):
        h, w = x.shape[-2:]

        layer_outputs = []
        for (get_overlap_patches, overlap_embed, layers) in self.stages:
            x = get_overlap_patches(x)

            num_patches = x.shape[-1]
            ratio = int(sqrt((h * w) / num_patches))
            x = rearrange(x, 'b c (h w) -> b c h w', h = h // ratio)

            x = overlap_embed(x)
            for (attn, ff) in layers:
                x = attn(x) + x
                x = ff(x) + x
            layer_outputs.append(x)
#         ret = x 
        if not return_layer_outputs:
            return x
        else: 
            return x, layer_outputs

segformer_type_args = {
    'segformer_b0': {
        'dims': (32, 64, 160, 256),      # dimensions of each stage
        'heads' : (1, 2, 5, 8),           # heads of each stage
        'ff_expansion' : (8, 8, 4, 4),    # feedforward expansion factor of each stage
        'reduction_ratio' : (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        'num_layers' : 2,                 # num layers of each stage
        'decoder_dim' : 256,              # decoder dimension
    }, 
    'segformer_b1': {
        'dims': (64, 128, 320, 512),      # dimensions of each stage
        'heads' : (1, 2, 5, 8),           # heads of each stage
        'ff_expansion' : (8, 8, 4, 4),    # feedforward expansion factor of each stage
        'reduction_ratio' : (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        'num_layers' : 2,                 # num layers of each stage
        'decoder_dim' : 256,              # decoder dimension
    },
    'segformer_b2': {
        'dims': (64, 128, 320, 512),      # dimensions of each stage
        'heads' : (1, 2, 5, 8),           # heads of each stage
        'ff_expansion' : (8, 8, 4, 4),    # feedforward expansion factor of each stage
        'reduction_ratio' : (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        'num_layers' : (3, 3, 6, 3),                 # num layers of each stage
        'decoder_dim' : 256,              # decoder dimension
    },
    'segformer_b3': {
        'dims': (64, 128, 320, 512),      # dimensions of each stage
        'heads' : (1, 2, 5, 8),           # heads of each stage
        'ff_expansion' : (8, 8, 4, 4),    # feedforward expansion factor of each stage
        'reduction_ratio' : (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        'num_layers' : (3, 3, 18, 3),                 # num layers of each stage
        'decoder_dim' : 256,              # decoder dimension
    },
    'segformer_b4': {
        'dims': (64, 128, 320, 512),      # dimensions of each stage
        'heads' : (1, 2, 5, 8),           # heads of each stage
        'ff_expansion' : (8, 8, 4, 4),    # feedforward expansion factor of each stage
        'reduction_ratio' : (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        'num_layers' : (3, 8, 27, 3),                 # num layers of each stage
        'decoder_dim' : 256,              # decoder dimension
    },
    'segformer_b5': {
        'dims': (64, 128, 320, 512),      # dimensions of each stage
        'heads' : (1, 2, 5, 8),           # heads of each stage
        'ff_expansion' : (4, 4, 4, 4),    # feedforward expansion factor of each stage
        'reduction_ratio' : (8, 4, 2, 1), # reduction ratio of each stage for efficient attention
        'num_layers' : (3, 6, 40, 3),                 # num layers of each stage
        'decoder_dim' : 256,              # decoder dimension
    }
}


class SegFormer(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 4
    ):
        super(SegFormer, self).__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

        self.mit = MiT(
            channels = channels,
            dims = dims,
            heads = heads,
            ff_expansion = ff_expansion,
            reduction_ratio = reduction_ratio,
            num_layers = num_layers
        )

#         self.mit.load_state_dict(torch.load('./pretrained/mit_b2.pth'))
        
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])

        self.to_segmentation = nn.Sequential(
            nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
            nn.Conv2d(decoder_dim, num_classes, 1),
        )

    def forward(self, x):
        out_size = x.shape[2:]
        
        layer_outputs = self.mit(x, return_layer_outputs = True)

        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
        fused = self.to_segmentation(fused).contiguous()
        fused = F.interpolate(
            fused, size=out_size, mode="bilinear", align_corners=False
        ) 
        return fused
    
class SegFormer_Body(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 4,
        return_attn = False
    ):
        super(SegFormer_Body, self).__init__()
        dims, heads, ff_expansion, reduction_ratio, num_layers = map(partial(cast_tuple, depth = 4), (dims, heads, ff_expansion, reduction_ratio, num_layers))
        assert all([*map(lambda t: len(t) == 4, (dims, heads, ff_expansion, reduction_ratio, num_layers))]), 'only four stages are allowed, all keyword arguments must be either a single value or a tuple of 4 values'

#         self.mit = MiT(
#             channels = channels,
#             dims = dims,
#             heads = heads,
#             ff_expansion = ff_expansion,
#             reduction_ratio = reduction_ratio,
#             num_layers = num_layers
#         )
        self.mit = mit_b2(return_attn=return_attn)

        self.mit.load_state_dict(torch.load('./pretrained/mit_b2.pth'), strict=False)

    def forward(self, x):
        return self.mit(x)


    
class SegFormer_Head(nn.Module):
    def __init__(
        self,
        *,
        dims = (32, 64, 160, 256),
        heads = (1, 2, 5, 8),
        ff_expansion = (8, 8, 4, 4),
        reduction_ratio = (8, 4, 2, 1),
        num_layers = 2,
        channels = 3,
        decoder_dim = 256,
        num_classes = 4,
        return_attn = False
    ):
        super(SegFormer_Head, self).__init__()
        
        self.to_fused = nn.ModuleList([nn.Sequential(
            nn.Conv2d(dim, decoder_dim, 1),
            nn.Upsample(scale_factor = 2 ** i)
        ) for i, dim in enumerate(dims)])
        
#         self.to_segmentation = nn.Sequential(
#             nn.Conv2d(4 * decoder_dim, decoder_dim, 1),
#             nn.Conv2d(decoder_dim, num_classes, 1),
#         )
        self.to_segmentation = nn.Conv2d(4 * decoder_dim, decoder_dim, 1)

    def forward(self, layer_outputs):
        fused = [to_fused(output) for output, to_fused in zip(layer_outputs, self.to_fused)]
        fused = torch.cat(fused, dim = 1)
#         return fused
        return self.to_segmentation(fused).contiguous(), layer_outputs