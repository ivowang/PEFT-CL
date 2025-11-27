""" Vision Transformer (ViT) with CPT (Continual Prompt Tuning)
Based on L2P implementation but adapted for CPT method
"""
import math
import logging
import os
import inspect
from functools import partial
from collections import OrderedDict
from typing import Optional
from urllib.parse import urlparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, resolve_pretrained_cfg, named_apply, adapt_input_conv, checkpoint_seq
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from timm.models.registry import register_model
import timm

from backbone.prompt import Prompt

_logger = logging.getLogger(__name__)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


def _get_pretrained_cfg_value(cfg, key, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _download_pretrained_npz(pretrained_url: str) -> str:
    """Download NPZ checkpoints and reuse cached copies."""
    if not pretrained_url:
        raise ValueError("Empty url for pretrained checkpoint.")

    download_cached_file = None
    try:
        from timm.models._hub import download_cached_file as timm_cached
        download_cached_file = timm_cached
    except Exception:
        try:
            from timm.models.hub import download_cached_file as timm_cached
            download_cached_file = timm_cached
        except Exception:
            download_cached_file = None

    if download_cached_file is not None:
        return download_cached_file(pretrained_url)

    cache_dir = os.path.join(torch.hub.get_dir(), 'checkpoints')
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(urlparse(pretrained_url).path) or 'checkpoint.npz'
    dest_path = os.path.join(cache_dir, filename)
    if not os.path.exists(dest_path):
        _logger.info(f"Downloading pretrained weights from {pretrained_url} to {dest_path}")
        torch.hub.download_url_to_file(pretrained_url, dest_path, progress=False)
    else:
        _logger.info(f"Using cached pretrained weights at {dest_path}")
    return dest_path


def _load_npz_weights(model, variant: str):
    """Helper to load NPZ weights defined in default_cfgs into a model."""
    pretrained_cfg = default_cfgs.get(variant, {})
    pretrained_url = pretrained_cfg.get('url', '')
    if not pretrained_url:
        _logger.warning(f"No pretrained url configured for {variant}")
        return
    if pretrained_url.endswith('.npz'):
        checkpoint_path = _download_pretrained_npz(pretrained_url)
        model.load_pretrained(checkpoint_path)
        _logger.info(f"Loaded NPZ weights for {variant} from {checkpoint_path}")
    else:
        _logger.warning(f"Pretrained url for {variant} is not an NPZ archive.")


default_cfgs = {
    'vit_base_patch16_224_cpt': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz'),
}


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values=None,
            drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class CPTPrompt(nn.Module):
    """CPT Prompt module supporting first_prompt, prompt, and meta_prompt"""
    def __init__(self, length=5, embed_dim=768, embedding_key='mean', prompt_init='uniform', 
                 prompt_pool=False, prompt_key=False, pool_size=None, top_k=None, 
                 batchwise_prompt=False, prompt_key_init='uniform', num_tasks=None):
        super().__init__()
        self.length = length
        self.embed_dim = embed_dim
        self.prompt_pool = prompt_pool
        self.embedding_key = embedding_key
        self.prompt_init = prompt_init
        self.prompt_key = prompt_key
        self.pool_size = pool_size
        self.top_k = top_k
        self.batchwise_prompt = batchwise_prompt
        self.num_tasks = num_tasks
        
        # CPT specific: separate prompts for first task, regular tasks, and meta-prompt
        if self.prompt_pool and pool_size is not None:
            # First prompt (for task 0)
            first_prompt_shape = (length, embed_dim)
            if prompt_init == 'zero':
                self.first_prompt = nn.Parameter(torch.zeros(first_prompt_shape))
            elif prompt_init == 'uniform':
                self.first_prompt = nn.Parameter(torch.randn(first_prompt_shape))
                nn.init.uniform_(self.first_prompt, -1, 1)
            
            # Regular prompts (one per task, stored in pool)
            prompt_pool_shape = (pool_size, length, embed_dim)
            if prompt_init == 'zero':
                self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
            elif prompt_init == 'uniform':
                self.prompt = nn.Parameter(torch.randn(prompt_pool_shape))
                nn.init.uniform_(self.prompt, -1, 1)
            
            # Meta-prompt (for initializing next task's prompt)
            meta_prompt_shape = (length, embed_dim)
            if prompt_init == 'zero':
                self.meta_prompt = nn.Parameter(torch.zeros(meta_prompt_shape))
            elif prompt_init == 'uniform':
                self.meta_prompt = nn.Parameter(torch.randn(meta_prompt_shape))
                nn.init.uniform_(self.meta_prompt, -1, 1)
        
        # Prompt keys
        if prompt_key:
            if prompt_pool:
                key_shape = (pool_size, embed_dim)
            else:
                key_shape = (1, embed_dim)
            if prompt_key_init == 'zero':
                self.prompt_key = nn.Parameter(torch.zeros(key_shape))
            elif prompt_key_init == 'uniform':
                self.prompt_key = nn.Parameter(torch.randn(key_shape))
                nn.init.uniform_(self.prompt_key, -1, 1)
        else:
            # Use mean of prompts as key (non-learnable)
            if prompt_pool:
                # Use mean of prompts as key
                with torch.no_grad():
                    prompt_mean = torch.mean(self.prompt, dim=1)
                self.prompt_key = prompt_mean
            else:
                # Use mean of first_prompt as key
                with torch.no_grad():
                    prompt_mean = torch.mean(self.first_prompt, dim=0, keepdim=True)
                self.prompt_key = prompt_mean
        
        self.current_task = -1
        self.training_mode = 'first_prompt'  # 'first_prompt', 'prompt', 'meta_prompt'
    
    def set_training_mode(self, mode):
        """Set training mode: 'first_prompt', 'prompt', or 'meta_prompt'"""
        self.training_mode = mode
    
    def set_current_task(self, task_id):
        """Set current task ID"""
        self.current_task = task_id
    
    def initialize_meta_prompt_from_prompt(self, task_id):
        """Initialize meta-prompt from a trained prompt of a specific task"""
        if self.prompt_pool and task_id < self.pool_size:
            with torch.no_grad():
                self.meta_prompt.data.copy_(self.prompt[task_id].data)
    
    def initialize_prompt_from_meta(self, task_id):
        """Initialize prompt for a new task from meta-prompt"""
        if self.prompt_pool and task_id < self.pool_size:
            with torch.no_grad():
                self.prompt[task_id].data.copy_(self.meta_prompt.data)
    
    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm
    
    def forward(self, x_embed, prompt_mask=None, cls_features=None, task_id=-1):
        out = dict()
        
        if self.prompt_pool:
            # Determine which prompt to use based on training mode
            if self.training_mode == 'first_prompt':
                # Use first_prompt for task 0
                batched_prompt = self.first_prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
                out['prompt_idx'] = None
            elif self.training_mode == 'meta_prompt':
                # Use meta_prompt
                batched_prompt = self.meta_prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
                out['prompt_idx'] = None
            else:
                # Use regular prompt pool with selection
                if self.embedding_key == 'mean':
                    x_embed_mean = torch.mean(x_embed, dim=1)
                elif self.embedding_key == 'max':
                    x_embed_mean = torch.max(x_embed, dim=1)[0]
                elif self.embedding_key == 'cls':
                    if cls_features is None:
                        x_embed_mean = torch.max(x_embed, dim=1)[0]
                    else:
                        x_embed_mean = cls_features
                else:
                    raise NotImplementedError("Not supported way of calculating embedding keys!")

                prompt_norm = self.l2_normalize(self.prompt_key, dim=1)
                x_embed_norm = self.l2_normalize(x_embed_mean, dim=1)

                similarity = torch.matmul(x_embed_norm, prompt_norm.t())
                
                if prompt_mask is None:
                    _, idx = torch.topk(similarity, k=self.top_k, dim=1)
                    if self.batchwise_prompt:
                        prompt_id, id_counts = torch.unique(idx, return_counts=True, sorted=True)
                        if prompt_id.shape[0] < self.pool_size:
                            prompt_id = torch.cat([prompt_id, torch.full((self.pool_size - prompt_id.shape[0],), torch.min(idx.flatten()), device=prompt_id.device)])
                            id_counts = torch.cat([id_counts, torch.full((self.pool_size - id_counts.shape[0],), 0, device=id_counts.device)])
                        _, major_idx = torch.topk(id_counts, k=self.top_k)
                        major_prompt_id = prompt_id[major_idx]
                        idx = major_prompt_id.expand(x_embed.shape[0], -1)
                else:
                    idx = prompt_mask

                batched_prompt_raw = self.prompt[idx]
                batch_size, top_k, length, c = batched_prompt_raw.shape
                batched_prompt = batched_prompt_raw.reshape(batch_size, top_k * length, c)

                out['prompt_idx'] = idx
                out['similarity'] = similarity
        else:
            # No prompt pool, use single prompt
            if self.training_mode == 'first_prompt':
                batched_prompt = self.first_prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
            elif self.training_mode == 'meta_prompt':
                batched_prompt = self.meta_prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
            else:
                # Use first_prompt as default
                batched_prompt = self.first_prompt.unsqueeze(0).expand(x_embed.shape[0], -1, -1)
        
        out['total_prompt_len'] = batched_prompt.shape[1]
        out['prompted_embedding'] = torch.cat([batched_prompt, x_embed], dim=1)
        
        return out


class VisionTransformer(nn.Module):
    """ Vision Transformer with CPT
    """
    def __init__(
            self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, global_pool='token',
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, init_values=None,
            class_token=True, no_embed_class=False, fc_norm=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            weight_init='', embed_layer=PatchEmbed, norm_layer=None, act_layer=None, block_fn=Block,
            prompt_length=None, embedding_key='cls', prompt_init='uniform', prompt_pool=False, prompt_key=False, pool_size=None,
            top_k=None, batchwise_prompt=False, prompt_key_init='uniform', head_type='token', use_prompt_mask=False,
            num_tasks=None):
        super().__init__()
        assert global_pool in ('', 'avg', 'token')
        assert class_token or global_pool != 'token'
        use_fc_norm = global_pool == 'avg' if fc_norm is None else fc_norm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.img_size = img_size
        self.num_classes = num_classes
        self.global_pool = global_pool
        self.num_features = self.embed_dim = embed_dim
        self.class_token = class_token
        self.num_prefix_tokens = 1 if class_token else 0
        self.no_embed_class = no_embed_class
        self.grad_checkpointing = False

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if class_token else None
        embed_len = num_patches if no_embed_class else num_patches + self.num_prefix_tokens
        # For CPT, we need to account for the maximum possible prompt length
        # first_prompt and meta_prompt use length, while prompt pool uses length * top_k
        if prompt_length is not None and pool_size is not None and prompt_pool:
            # Use the maximum: either prompt_length (for first/meta) or prompt_length * top_k (for pool)
            max_prompt_len = max(prompt_length, prompt_length * top_k) if top_k else prompt_length
            embed_len += max_prompt_len
        self.pos_embed = nn.Parameter(torch.randn(1, embed_len, embed_dim) * .02)
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.prompt_pool = prompt_pool
        self.head_type = head_type
        self.use_prompt_mask = use_prompt_mask
        
        if prompt_length is not None and pool_size is not None and prompt_pool: 
            self.prompt = CPTPrompt(
                length=prompt_length, embed_dim=embed_dim, embedding_key=embedding_key, 
                prompt_init=prompt_init, prompt_pool=prompt_pool, prompt_key=prompt_key, 
                pool_size=pool_size, top_k=top_k, batchwise_prompt=batchwise_prompt,
                prompt_key_init=prompt_key_init, num_tasks=num_tasks)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.Sequential(*[
            block_fn(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, init_values=init_values,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim) if not use_fc_norm else nn.Identity()

        # Classifier Head
        self.fc_norm = norm_layer(embed_dim) if use_fc_norm else nn.Identity()
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if weight_init != 'skip':
            self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'moco', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
        named_apply(get_init_weights_vit(mode, head_bias), self)

    def _init_weights(self, m):
        init_weights_vit_timm(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^cls_token|pos_embed|patch_embed',
            blocks=[(r'^blocks\.(\d+)', None), (r'^norm', (99999,))]
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes: int, global_pool=None):
        self.num_classes = num_classes
        if global_pool is not None:
            assert global_pool in ('', 'avg', 'token')
            self.global_pool = global_pool
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, task_id=-1, cls_features=None, train=False):
        x = self.patch_embed(x)

        if hasattr(self, 'prompt'):
            if self.use_prompt_mask and train:
                start = task_id * self.prompt.top_k
                end = (task_id + 1) * self.prompt.top_k
                single_prompt_mask = torch.arange(start, end).to(x.device)
                prompt_mask = single_prompt_mask.unsqueeze(0).expand(x.shape[0], -1)
                if end > self.prompt.pool_size:
                    prompt_mask = None
            else:
                prompt_mask = None
            res = self.prompt(x, prompt_mask=prompt_mask, cls_features=cls_features, task_id=task_id)
            self.total_prompt_len = res['total_prompt_len']
            x = res['prompted_embedding']
        else:
            res = dict()
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        # Adjust pos_embed to match actual sequence length
        seq_len = x.shape[1]
        pos_embed_len = self.pos_embed.shape[1]
        if seq_len != pos_embed_len:
            if seq_len < pos_embed_len:
                # Truncate pos_embed if sequence is shorter
                pos_embed = self.pos_embed[:, :seq_len, :]
            else:
                # Pad pos_embed if sequence is longer (shouldn't happen normally)
                # Use the last position embedding for padding
                padding = self.pos_embed[:, -1:, :].expand(1, seq_len - pos_embed_len, -1)
                pos_embed = torch.cat([self.pos_embed, padding], dim=1)
        else:
            pos_embed = self.pos_embed
        
        x = self.pos_drop(x + pos_embed)

        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x)
        else:
            x = self.blocks(x)
        
        x = self.norm(x)
        res['x'] = x

        return res

    def forward_head(self, res, pre_logits: bool = False):
        x = res['x']
        if self.class_token and self.head_type == 'token':
            x = x[:, 0]
        elif self.head_type == 'gap' and self.global_pool == 'avg':
            x = x.mean(dim=1)
        elif self.head_type == 'prompt' and self.prompt_pool:
            x = x[:, 1:(1 + self.total_prompt_len)] if self.class_token else x[:, 0:self.total_prompt_len]
            x = x.mean(dim=1)
        elif self.head_type == 'token+prompt' and self.prompt_pool and self.class_token:
            x = x[:, 0:self.total_prompt_len + 1]
            x = x.mean(dim=1)
        else:
            raise ValueError(f'Invalid head_type={self.head_type}')
        
        res['pre_logits'] = x

        x = self.fc_norm(x)
        
        res['logits'] = self.head(x)
        
        return res

    def forward(self, x, task_id=-1, cls_features=None, train=False):
        res = self.forward_features(x, task_id=task_id, cls_features=cls_features, train=train)
        res = self.forward_head(res)
        return res


def init_weights_vit_timm(module: nn.Module, name: str = ''):
    """ ViT weight initialization, original timm impl (for reproducibility) """
    if isinstance(module, nn.Linear):
        trunc_normal_(module.weight, std=.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def init_weights_vit_jax(module: nn.Module, name: str = '', head_bias: float = 0.):
    """ ViT weight initialization, matching JAX (Flax) impl """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        else:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=1e-6) if 'mlp' in name else nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif hasattr(module, 'init_weights'):
        module.init_weights()


def get_init_weights_vit(mode='jax', head_bias: float = 0.):
    if 'jax' in mode:
        return partial(init_weights_vit_jax, head_bias=head_bias)
    else:
        return init_weights_vit_timm


@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(
            pos_embed_w,
            model.pos_embed,
            getattr(model, 'num_prefix_tokens', 1),
            model.patch_embed.grid_size
        )
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    if isinstance(model.head, nn.Linear) and model.head.bias.shape[0] == w[f'{prefix}head/bias'].shape[-1]:
        model.head.weight.copy_(_n2p(w[f'{prefix}head/kernel']))
        model.head.bias.copy_(_n2p(w[f'{prefix}head/bias']))


def resize_pos_embed(posemb, posemb_new, num_prefix_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_prefix_tokens:
        posemb_prefix, posemb_grid = posemb[:, :num_prefix_tokens], posemb[0, num_prefix_tokens:]
    else:
        posemb_prefix, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if ntok_new > gs_old ** 2:
        ntok_new -= gs_old ** 2
        posemb_prefix = posemb_prefix.expand(-1, ntok_new, -1)
    if not len(gs_new):
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_prefix, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model, adapt_layer_scale=False):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    import re
    out_dict = {}
    if 'model' in state_dict:
        state_dict = state_dict['model']

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape[1] != model.pos_embed.shape[1]:
            v = resize_pos_embed(
                v,
                model.pos_embed,
                0 if getattr(model, 'no_embed_class') else getattr(model, 'num_prefix_tokens', 1),
                model.patch_embed.grid_size
            )
        elif adapt_layer_scale and 'gamma_' in k:
            k = re.sub(r'gamma_([0-9])', r'ls\1.gamma', k)
        elif 'pre_logits' in k:
            continue
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    pretrained_cfg = resolve_pretrained_cfg(variant, pretrained_cfg=kwargs.pop('pretrained_cfg', None))
    pretrained_url = _get_pretrained_cfg_value(pretrained_cfg, 'url', '') or ''
    build_args = dict(
        pretrained_cfg=pretrained_cfg,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_strict=False,
    )
    builder_params = inspect.signature(build_model_with_cfg).parameters
    if 'pretrained_custom_load' in builder_params:
        build_args['pretrained_custom_load'] = 'npz' in pretrained_url

    model = build_model_with_cfg(
        VisionTransformer, variant, False,
        **build_args,
        **kwargs)

    if pretrained:
        try:
            if pretrained_url.endswith('.npz'):
                checkpoint_path = _download_pretrained_npz(pretrained_url)
                model.load_pretrained(checkpoint_path)
                _logger.info(f"Successfully loaded pretrained NPZ weights for {variant} from {checkpoint_path}")
            else:
                _logger.info(f"Loading pretrained weights for {variant} using timm...")
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        from timm.models import vision_transformer as timm_vit
                        if hasattr(timm_vit, variant):
                            pretrained_model = getattr(timm_vit, variant)(pretrained=True, num_classes=0)
                        else:
                            raise AttributeError(f"{variant} not found in vision_transformer")
                    except Exception as e1:
                        _logger.warning(f"Direct import failed ({e1}), trying create_model...")
                        pretrained_model = timm.create_model(variant, pretrained=True, num_classes=0)
                pretrained_state_dict = pretrained_model.state_dict()
                del pretrained_model
                model_state_dict = model.state_dict()
                filtered_state_dict = {}
                for key, value in pretrained_state_dict.items():
                    if 'head' in key or 'classifier' in key:
                        continue
                    if key in model_state_dict and model_state_dict[key].shape == value.shape:
                        filtered_state_dict[key] = value
                missing_keys, unexpected_keys = model.load_state_dict(filtered_state_dict, strict=False)
                if missing_keys:
                    non_prompt_missing = [k for k in missing_keys if 'prompt' not in k.lower()]
                    if non_prompt_missing:
                        _logger.warning(f"Missing keys (non-prompt): {non_prompt_missing[:5]}...")
                if unexpected_keys:
                    _logger.debug(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
                _logger.info(f"Successfully loaded {len(filtered_state_dict)} pretrained weights for {variant}")
        except Exception as err:
            import traceback
            _logger.error(
                f"Failed to load pretrained weights for {variant}: {err}\n"
                f"Traceback: {traceback.format_exc()}\n"
                f"Continuing with randomly initialized weights."
            )
    return model


def create_cpt_teacher(
        pretrained: bool = True,
        num_classes: int = 1000,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        **kwargs):
    """Create a plain ViT teacher (without CPT prompts) initialized from NPZ weights."""
    teacher = VisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        num_classes=num_classes,
        global_pool='token',
        head_type='token',
        embed_dim=kwargs.get('embed_dim', 768),
        depth=kwargs.get('depth', 12),
        num_heads=kwargs.get('num_heads', 12),
        mlp_ratio=kwargs.get('mlp_ratio', 4.0),
        qkv_bias=kwargs.get('qkv_bias', True),
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        prompt_pool=False,
        prompt_length=None,
        prompt_key=False,
        pool_size=None,
        top_k=None,
    )
    if pretrained:
        _load_npz_weights(teacher, 'vit_base_patch16_224')
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


@register_model
def vit_base_patch16_224_cpt(pretrained=False, **kwargs):
    """ ViT-Base (ViT-B/16) with CPT
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer('vit_base_patch16_224_cpt', pretrained=pretrained, **model_kwargs)
    return model

