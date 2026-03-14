import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import gin
import os
import pandas as pd
import torchaudio
from typing import Callable

import gamadhani.utils.pitch_to_audio_utils as p2a
from gamadhani.utils.utils import prob_mask_like
from x_transformers.x_transformers import AttentionLayers
import pdb

def get_activation(act: str = 'mish'):
    act = act.lower()
    if act == 'mish':
        return nn.Mish()
    elif act == 'relu':
        return nn.ReLU()
    elif act == 'leaky_relu':
        return nn.LeakyReLU()
    elif act == 'gelu':
        return nn.GELU()
    elif act == 'swish':
        return nn.SiLU()
    else:
        raise ValueError(f'Activation {act} not supported')
    
def get_weight_norm(layer):
    return torch.nn.utils.parametrizations.weight_norm(layer)
   
def get_layer(layer, norm: bool):
    if norm:
        return get_weight_norm(layer)
    else:
        return layer

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super(PositionalEncoding, self).__init__()
        self.dim = dim

    def forward(self, x):
        shape = x.shape
        x = x * 100
        w = torch.pow(10000, (2 * torch.arange(self.dim // 2).float() / self.dim)).to(x)
        x = x.unsqueeze(-1) / w
        embed = torch.cat([torch.cos(x), torch.sin(x)], -1)
        embed = embed.reshape(*shape, -1)
        if len(shape) == 2:  # f0 embedding, else time embedding
            embed = embed.permute(0, 2, 1)
        return embed

class ConvBlock(nn.Module):
    def __init__(self, 
                 inp_dim, 
                 out_dim, 
                 kernel_size: int = 3,
                 stride: int = 1, 
                 padding: Union[str, int] = "same",
                 norm: bool = True,
                 nonlinearity: Optional[str] = None,
                 up: bool = False,
                 dropout: float = 0.0,
                 ):
        super(ConvBlock, self).__init__()
        self.inp_dim = inp_dim
        self.out_dim = out_dim
        # self.norm = norm
        # pdb.set_trace()
        if nonlinearity is not None:
            self.nonlinearity = get_activation(nonlinearity)
        else:
            self.nonlinearity = None
        if up:
            self.conv = get_layer(nn.ConvTranspose1d(inp_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding), norm)
        else:
            self.conv = get_layer(nn.Conv1d(inp_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding), norm)

        self.layers = nn.ModuleList()
        if self.nonlinearity is not None:
            self.layers.append(self.nonlinearity)
        if dropout > 0:
            self.layers.append(nn.Dropout(dropout))
        self.layers.append(self.conv)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
class UpSampleLayer(nn.Module):
    def __init__(self, 
                inp_dim, 
                out_dim, 
                kernel_size: int = 3,
                stride: int = 1, 
                padding: Union[str, int] = "same",
                num_convs: int = 2,
                norm: bool = True,
                nonlinearity: Optional[str] = None,
                dropout: float = 0.0,
                ):
        super(UpSampleLayer, self).__init__()
        assert num_convs > 0, "Number of convolutions must be greater than 0"
        self.num_convs = num_convs

        self.convs = nn.ModuleList([])

        self.convs.append(ConvBlock(inp_dim, out_dim, kernel_size=stride*2, stride=stride, padding=padding, norm=norm, nonlinearity=nonlinearity, up=True))  # first convolutional layer to upsample
        for ind in range(1, num_convs):
            self.convs.append(ConvBlock(out_dim, out_dim, kernel_size=kernel_size, stride=1, padding="same", norm=norm, nonlinearity=nonlinearity, up=False, dropout=dropout if ind == num_convs-1 else 0))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x
    
class DownSampleLayer(nn.Module):
    def __init__(self, 
                inp_dim, 
                out_dim, 
                kernel_size: int = 3,
                stride: int = 1, 
                padding: Union[str, int] = "same",
                num_convs: int = 2,
                norm: bool = True,
                nonlinearity: Optional[str] = None,
                dropout: float = 0.0,
                ):
        super(DownSampleLayer, self).__init__()
        assert num_convs > 0, "Number of convolutions must be greater than 0"
        self.num_convs = num_convs

        self.convs = nn.ModuleList([])

        self.convs.append(ConvBlock(inp_dim, out_dim, kernel_size=stride*2, stride=stride, padding=padding, norm=norm, nonlinearity=nonlinearity, up=False))  # first convolutional layer to upsample
        for ind in range(1, num_convs):
            self.convs.append(ConvBlock(out_dim, out_dim, kernel_size=kernel_size, stride=1, padding="same", norm=norm, nonlinearity=nonlinearity, up=False, dropout=dropout if ind == num_convs-1 else 0))

    def forward(self, x):
        for conv in self.convs:
            x = conv(x)
        return x

# class Attention(nn.Module):
#     def __init__(self, 
#                  num_heads, 
#                  num_channels,
#                  dropout=0.0):
#         super(Attention, self).__init__()
#         self.num_heads = num_heads
#         self.num_channels = num_channels
#         self.layer_norm1 = nn.LayerNorm(self.num_channels)
#         self.layer_norm2 = nn.LayerNorm(self.num_channels)
#         self.qkv_proj = nn.Linear(self.num_channels, self.num_channels * 3, bias=False)
#         self.head_dim = self.num_channels // self.num_heads
#         self.final_proj = nn.Linear(self.num_channels, self.num_channels)
#         self.dropout = nn.Dropout(dropout)

#     def split_heads(self, x):
#         # input shape bs, time, channels
#         x = x.view(x.shape[0], x.shape[1], self.num_heads, self.head_dim)
#         return x.permute(0, 2, 1, 3) # bs, num_heads, time, head_dim
    
#     def forward(self, x):
#         # pdb.set_trace()
#         x = torch.permute(x, (0, 2, 1)) # bs, time, channels
#         residual = x
#         x = self.layer_norm1(x)
#         x = self.qkv_proj(x)
#         q, k, v = x.chunk(3, dim=-1)

#         # split heads
#         q = self.split_heads(q)
#         k = self.split_heads(k)
#         v = self.split_heads(v)

#         # calculate attention
#         x = torch.einsum("...td,...sd->...ts", q, k) / math.sqrt(self.head_dim)
#         x = self.dropout(x)
#         x = torch.einsum("...ts,...sd->...td", F.softmax(x, dim=-1), v) # bs, num_heads, time, head_dim

#         # combine heads
#         x = torch.permute(x, (0, 2, 1, 3)) # bs, time, num_heads, head_dim
#         x = x.reshape(x.shape[0], x.shape[1], self.num_heads * self.head_dim)

#         # final projection
#         x = self.final_proj(x)
#         x = self.layer_norm2(x + residual)
#         return torch.permute(x, (0, 2, 1)) # bs, channels, time

class ResNetBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 dropout: float = 0.0,
                 nonlinearity: Optional[str] = None,
                 kernel_size: int = 3,
                 stride: int = 1,
                 norm: bool = True,
                up: bool = False,
                num_convs: int = 2,
                ):
        super(ResNetBlock, self).__init__()
        
        self.input_layers = nn.ModuleList([])
        if nonlinearity is not None:
            self.input_layers.append(get_activation(nonlinearity))
        
        if up:
            self.input_layers.append(get_layer(nn.ConvTranspose1d(in_channels, out_channels, kernel_size=stride*2, stride=stride, padding=stride//2), norm))
        else:
            if in_channels != out_channels:
                self.input_layers.append(get_layer(nn.Conv1d(in_channels, out_channels, kernel_size=stride*2, stride=stride, padding=stride//2), norm))
            elif stride > 1:
                self.input_layers.append(nn.AvgPool1d(stride*2, stride=stride, padding=stride//2))
            else:
                self.input_layers.append(nn.Identity())

        if up:
            self.process_layer = UpSampleLayer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=stride//2, num_convs=num_convs, norm=norm, nonlinearity=nonlinearity, dropout=dropout)
        else:
            self.process_layer = DownSampleLayer(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=stride//2, num_convs=num_convs, norm=norm, nonlinearity=nonlinearity, dropout=dropout)

    def forward(self, x):
        # pdb.set_trace()
        inputs = x.clone()
        for layer in self.input_layers:
            inputs = layer(inputs)
        x = self.process_layer(x)
        return x + inputs
            
@gin.configurable
class UNetBase(pl.LightningModule):
    def __init__(self, log_grad_norms_every=10):
        super(UNetBase, self).__init__()
        self.log_grad_norms_every = log_grad_norms_every

    @gin.configurable
    def configure_optimizers(self, optimizer_cls: Callable[[], torch.optim.Optimizer],
    scheduler_cls: Callable[[],
                            torch.optim.lr_scheduler._LRScheduler]):
        # pdb.set_trace()
        optimizer = optimizer_cls(self.parameters())
        scheduler = scheduler_cls(optimizer)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}] 

@gin.configurable
class UNet(UNetBase):
    def __init__(self,
                 inp_dim, 
                 time_dim, 
                 features, 
                 strides, 
                 kernel_size, 
                 seq_len, 
                 project_dim=None,
                 dropout=0.0,
                 nonlinearity=None,
                 norm=True,
                 num_convs=2,
                 num_attns=2,
                 num_heads=8,
                 log_samples_every=10, 
                 ckpt=None,
                 loss_w_padding=False,
                 groups=None,
                 nfft=None,
                log_grad_norms_every=10
                 ):
        super(UNet, self).__init__()
        self.time_dim = time_dim
        self.features = features
        self.strides = strides
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.log_samples_every = log_samples_every
        self.ckpt = ckpt
        self.strides_prod = np.prod(strides)
        self.loss_w_padding = loss_w_padding
        self.inp_dim = inp_dim

        if log_grad_norms_every is not None:    
            assert log_grad_norms_every > 0, "log_grad_norms_every must be greater than 0"
        self.log_grad_norms_every = log_grad_norms_every

        if project_dim is None:
            project_dim = features[0]
        self.initial_projection = nn.Conv1d(inp_dim, project_dim, kernel_size=1)
        self.positional_encoding = PositionalEncoding(time_dim)

        features = [project_dim] + features
        strides = [None] + strides

        self.downsample_layers = nn.ModuleList([
            ResNetBlock(features[ind-1] + time_dim, 
                        features[ind], 
                        kernel_size=kernel_size, 
                        stride=strides[ind],
                        dropout=dropout,
                        nonlinearity=nonlinearity,
                        norm=norm,
                        num_convs=num_convs,
                        ) for ind in range(1, len(features))
        ])
        
        # self.attention_layers = nn.ModuleList(
        #     [Attention(num_heads=num_heads, num_channels=features[-1], dropout=dropout) for _ in range(num_attns)]
        # )

        self.attention_layers = AttentionLayers(
            dim = features[-1],
            heads = num_heads,
            depth = num_attns,
        )

        self.upsample_layers = nn.ModuleList([
            ResNetBlock(features[ind] * 2 + time_dim, # input size defined by features + skip dimension + time dimension  
            features[ind-1], 
            kernel_size=kernel_size, 
            stride=strides[ind],
            dropout=dropout,
            nonlinearity=nonlinearity,
            norm=norm,
            num_convs=num_convs,
            up=True
            ) for ind in range(len(features) - 1, 0, -1)
        ])
        self.final_projection = nn.Conv1d(2*project_dim, inp_dim, kernel_size=1)
    
    def pad_to(self, x, strides):
        # modified from: https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
        l = x.shape[-1]

        if l % strides > 0:
            new_l = l + strides - l % strides
        else:
            new_l = l

        ll, ul = int((new_l-l) / 2), int(new_l-l) - int((new_l-l) / 2)
        pads = (ll, ul)

        out = F.pad(x, pads, "reflect").to(x)

        return out, pads

    def unpad(self, x, pad):
        # modified from: https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
        if pad[0]+pad[1] > 0:
            x = x[:,:,pad[0]:-pad[1]]
        return x

    def forward(self, x, time):
        
        # INITIAL PROJECTION
        x = self.initial_projection(x)

        # TIME CONDITIONING
        time = self.positional_encoding(time)

        def _concat_time(x, time):
            time = time.unsqueeze(2).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, time], -2)
            return x

        skips = []

        # DOWNSAMPLING
        for ind, downsample_layer in enumerate(self.downsample_layers):
            # print(f'Down sample layer {ind}')
            skips.append(x)
            x = _concat_time(x, time)
            x = downsample_layer(x)
        skips.append(x)

        # BOTTLENECK ATTENTION
        x = torch.permute(x, (0, 2, 1))
        x = self.attention_layers(x)
        x = torch.permute(x, (0, 2, 1))
        # pdb.set_trace()
        # UPSAMPLING
        for ind, upsample_layer in enumerate(self.upsample_layers):
            # print(f'Up sample layer {ind}')
            x = _concat_time(x, time)
            x = torch.cat([x, skips.pop(-1)], 1)
            x = upsample_layer(x)
        x = torch.cat([x, skips.pop(-1)], 1)

        # FINAL PROJECTION
        x = self.final_projection(x)
        return x

    def loss(self, x):
        # pdb.set_trace()
        padded_x, padding = self.pad_to(x, self.strides_prod)
        t = torch.rand((padded_x.shape[0],)).to(padded_x)
        noise = torch.normal(0, 1, padded_x.shape).to(padded_x)
        # print(t.device, noise.device, x.device)
        x_t = t[:, None, None] * padded_x + (1 - t[:, None, None]) * noise
        # print(t.device, noise.device, x_t.device, x.device)
        padded_y = self.forward(x_t, t)
        unpadded_y = self.unpad(padded_y, padding)

        if self.loss_w_padding:
            target = padded_x - noise
            return torch.mean((padded_y - target) ** 2)
        else:
            target = x - self.unpad(noise, padding) # x1 - x0
            return torch.mean((unpadded_y - target) ** 2)

    
    def on_before_optimizer_step(self, optimizer, *_):
        def calculate_grad_norm(module_list, norm_type=2):
            total_norm = 0
            if isinstance(module_list, nn.Module):
                module_list = [module_list]
            for module in module_list:
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        param_norm = torch.norm(param.grad.detach(), p=norm_type)
                        total_norm += param_norm**2
            # pdb.set_trace()
            total_norm = torch.sqrt(total_norm)
            return total_norm
        
        if self.log_grad_norms_every is not None and self.global_step % self.log_grad_norms_every == 0:
            self.log('Grad Norm/Downsample Layers', calculate_grad_norm(self.downsample_layers))
            self.log('Grad Norm/Attention Layers', calculate_grad_norm(self.attention_layers))
            self.log('Grad Norm/Upsample Layers', calculate_grad_norm(self.upsample_layers))

    def training_step(self, batch, batch_idx):
        # print('\n', batch_idx, batch.shape)
        x = batch 
        loss = self.loss(x)

        # log grad_norms
        # if self.log_grad_norms_every > 0 and self.current_epoch % self.log_grad_norms_every == 0:
        
        # for ind, attention_layer in enumerate(self.attention_layers):
        #     self.log(f'Grad Norm/Attention Layer {ind}', grad_norm(attention_layer, norm_type=2))
        # for ind, downsample_layer in enumerate(self.downsample_layers):
            # self.log(f'Grad Norm/Downsample Layer {ind}', grad_norm(downsample_layer, norm_type=2))

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self.loss(x)
        self.log('val_loss', loss)
        return loss
    
    def sample_fn(self, batch_size: int, num_steps: int, prime: Optional[torch.Tensor] = None):
        # CREATE INITIAL NOISE
        if prime is not None:
            prime = prime.to(self.device)
        noise = torch.normal(mean=0.0, std=1.0, size=(batch_size, 1, self.seq_len)).to(self.device)
        x_alpha_t = noise.clone()
        t_array = torch.ones((batch_size,)).to(self.device)
        # x_alpha_ts = {}
        with torch.no_grad():
            # SAMPLE FROM MODEL
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                t_tensor = torch.tensor(t)
                alpha_t = t_tensor * t_array 
                alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).to(self.device)
                if prime is not None:
                    x_alpha_t[:, :, :prime.shape[-1]] = ((1 - alpha_t) * noise[:, :, :prime.shape[-1]]) + (alpha_t * prime) # fill in the prime in the beginning of each x_t
                diff =  self.forward(x_alpha_t, t_tensor * t_array)
                x_alpha_t = x_alpha_t + 1 / num_steps * diff
        return x_alpha_t

    def sample_sdedit(self, cond, batch_size, num_steps, t0=0.5):
        t0_steps = int((1-t0)*num_steps) 
        t_array = torch.ones((batch_size,)).to(self.device)
        x_alpha_t = cond.clone() 
        with torch.no_grad():
            x_alpha_t = t0 * x_alpha_t + (1 - t0) * torch.normal(0, 1, x_alpha_t.shape).to(x_alpha_t)
            for t in np.linspace(t0, 1, t0_steps)[:-1]:
                t_tensor = torch.tensor(t)
                x_alpha_t = x_alpha_t + 1 / num_steps * self.forward(x_alpha_t, t_tensor * t_array)

        return x_alpha_t

    

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % self.log_samples_every == 0:
            samples = self.sample_fn(16, 100).detach().cpu().numpy()
            if self.ckpt is not None:
                os.makedirs(os.path.join(self.ckpt, 'samples', str(self.current_epoch)), exist_ok=True)
            fig, axs = plt.subplots(4, 4, figsize=(16, 16))
            for i in range(4):
                for j in range(4):
                    axs[i, j].plot(samples[i*4+j].squeeze())
                    pd.DataFrame(samples[i*4+j].squeeze(), columns=['normalized_pitch']).to_csv(os.path.join(self.ckpt, 'samples', str(self.current_epoch), f'sample_{i*4+j}.csv'))
            if self.logger:
                wandb.log({"samples": [wandb.Image(fig, caption="Samples")]})
            else:
                fig.savefig(os.path.join(self.ckpt, 'samples', str(self.current_epoch), 'samples.png'))
            plt.close(fig)


@gin.configurable
class UNetAudio(UNetBase):
    def __init__(self,
                 inp_dim, 
                 time_dim, 
                 features, 
                 strides, 
                 kernel_size, 
                 seq_len, 
                 project_dim=None,
                 dropout=0.0,
                 nonlinearity=None,
                 norm=True,
                 num_convs=2,
                 num_attns=2,
                 num_heads=8,
                 ckpt=None,
                 qt = None,
                 log_samples_every = 10,
                 log_wandb_samples_every = 50,
                 sr=16000,
                 loss_w_padding=False,
                 log_grad_norms_every=10
                 ):
        super(UNetAudio, self).__init__()
        self.inp_dim = inp_dim
        self.time_dim = time_dim
        self.features = features
        self.strides = strides
        self.kernel_size = kernel_size
        self.seq_len = seq_len
        self.log_samples_every = log_samples_every
        self.log_wandb_samples_every = log_wandb_samples_every
        self.ckpt = ckpt
        self.qt = qt
        self.sr = sr
        self.strides_prod = np.prod(strides)
        self.loss_w_padding = loss_w_padding
        self.log_grad_norms_every = log_grad_norms_every

        if project_dim is None:
            project_dim = features[0]
        self.initial_projection = nn.Conv1d(inp_dim, project_dim, kernel_size=1)
        self.positional_encoding = PositionalEncoding(time_dim)

        features = [project_dim] + features
        strides = [None] + strides

        self.downsample_layers = nn.ModuleList([
            ResNetBlock(features[ind-1] + time_dim, 
                        features[ind], 
                        kernel_size=kernel_size, 
                        stride=strides[ind],
                        dropout=dropout,
                        nonlinearity=nonlinearity,
                        norm=norm,
                        num_convs=num_convs,
                        ) for ind in range(1, len(features))
        ])
        
        self.attention_layers = AttentionLayers(
            dim = features[-1],
            heads = num_heads,
            depth = num_attns,
        )

        self.upsample_layers = nn.ModuleList([
            ResNetBlock(features[ind] * 2 + time_dim, # input size defined by features + skip dimension + time dimension  
            features[ind-1], 
            kernel_size=kernel_size, 
            stride=strides[ind],
            dropout=dropout,
            nonlinearity=nonlinearity,
            norm=norm,
            num_convs=num_convs,
            up=True
            ) for ind in range(len(features) - 1, 0, -1)
        ])
        self.final_projection = nn.Conv1d(2*project_dim, inp_dim, kernel_size=1)
        self.losses = []

    def forward(self, x, time):
        # INITIAL PROJECTION
        x = self.initial_projection(x)

        # TIME CONDITIONING
        time = self.positional_encoding(time)

        def _concat_time(x, time):
            time = time.unsqueeze(2).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, time], -2)
            return x

        skips = []

        # DOWNSAMPLING
        for ind, downsample_layer in enumerate(self.downsample_layers):
            # print(f'Down sample layer {ind}')
            skips.append(x)
            x = _concat_time(x, time)
            x = downsample_layer(x)
        skips.append(x)
        # BOTTLENECK ATTENTION
        x = torch.permute(x, (0, 2, 1))
        x = self.attention_layers(x)
        x = torch.permute(x, (0, 2, 1))

        # pdb.set_trace()
        # UPSAMPLING
        for ind, upsample_layer in enumerate(self.upsample_layers):
            # print(f'Up sample layer {ind}')
            x = _concat_time(x, time)
            x = torch.cat([x, skips.pop(-1)], 1)
            x = upsample_layer(x)
        x = torch.cat([x, skips.pop(-1)], 1)

        # FINAL PROJECTION
        x = self.final_projection(x)
        return x

    def pad_to(self, x, strides):
        # modified from: https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
        l = x.shape[-1]

        if l % strides > 0:
            new_l = l + strides - l % strides
        else:
            new_l = l

        ll, ul = int((new_l-l) / 2), int(new_l-l) - int((new_l-l) / 2)
        pads = (ll, ul)

        out = F.pad(x, pads, "reflect").to(x)

        return out, pads

    def unpad(self, x, pad):
        # modified from: https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
        if pad[0]+pad[1] > 0:
            x = x[:,:,pad[0]:-pad[1]]
        return x
    
    def loss(self, x):
        padded_x, padding = self.pad_to(x, self.strides_prod)
        t = torch.rand((padded_x.shape[0],)).to(padded_x)
        noise = torch.normal(0, 1, padded_x.shape).to(padded_x)
        # print(t.device, noise.device, x.device)
        x_t = t[:, None, None] * padded_x + (1 - t[:, None, None]) * noise
        # print(t.device, noise.device, x_t.device, x.device)
        padded_y = self.forward(x_t, t)
        unpadded_y = self.unpad(padded_y, padding)

        if self.loss_w_padding:
            target = padded_x - noise
            return torch.mean((padded_y - target) ** 2)
        else:
            target = x - self.unpad(noise, padding) # x1 - x0
            return torch.mean((unpadded_y - target) ** 2)

    def training_step(self, batch, batch_idx):
        # print('\n', batch_idx, batch.shape)
        x = batch 
        loss = self.loss(x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        loss = self.loss(x)
        self.log('val_loss', loss)
        return loss
    
    def sample_fn(self, batch_size: int, num_steps: int, prime=None):
        if prime is not None:
            prime = prime.to(self.device)
        # CREATE INITIAL NOISE
        noise = torch.normal(mean=0, std=1, size=(batch_size, self.inp_dim, self.seq_len)).to(self.device)
        padded_noise, padding = self.pad_to(noise, self.strides_prod)
        x_alpha_t = padded_noise.clone()
        t_array = torch.ones((batch_size,)).to(self.device)
        with torch.no_grad():
            # SAMPLE FROM MODEL
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                t_tensor = torch.tensor(t)
                alpha_t = t_tensor * t_array
                alpha_t = alpha_t.unsqueeze(1).unsqueeze(2).to(self.device)
                if prime is not None:
                    x_alpha_t[:, :, :prime.shape[-1]] = ((1 - alpha_t) * noise[:, :, :prime.shape[-1]]) + (alpha_t * prime) # fill in the prime in the beginning of each x_t
                diff =  self.forward(x_alpha_t, t_tensor * t_array)
                x_alpha_t = x_alpha_t + 1 / num_steps * diff
            
            padded_y = x_alpha_t
            unpadded_y = self.unpad(padded_y, padding)
        
        return unpadded_y

    def on_validation_epoch_end(self) -> None:
        if self.current_epoch % self.log_samples_every == 0:
            if self.ckpt is not None:
                os.makedirs(os.path.join(self.ckpt, 'samples', str(self.current_epoch)), exist_ok=True)
            samples = self.sample_fn(16, 100)
            audio = p2a.normalized_mels_to_audio(samples, qt=self.qt)
            beep = torch.sin(2 * torch.pi * 220 * torch.arange(0, 0.1 * self.sr) / self.sr).to(audio)
            concat_audios = []
            for sample in audio:
                concat_audios.append(torch.cat([sample, beep]))
            concat_audio = torch.cat(concat_audios, dim=-1).reshape(1, -1).to('cpu')
            output_file = os.path.join(self.ckpt, 'samples', f'samples_{self.current_epoch}.wav')
            torchaudio.save(output_file, concat_audio, self.sr)
            if self.current_epoch % self.log_wandb_samples_every == 0:
                if self.logger:
                    wandb.log({
                        "samples": [wandb.Audio(output_file, self.sr, caption="Samples")]})

    def on_before_optimizer_step(self, optimizer, *_):
        def calculate_grad_norm(module_list, norm_type=2):
            total_norm = 0
            if isinstance(module_list, nn.Module):
                module_list = [module_list]
            for module in module_list:
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        param_norm = torch.norm(param.grad.detach(), p=norm_type)
                        total_norm += param_norm**2
            # pdb.set_trace()
            total_norm = torch.sqrt(total_norm)
            return total_norm
    
        if self.log_grad_norms_every is not None and self.global_step % self.log_grad_norms_every == 0:
            self.log('Grad Norm/Downsample Layers', calculate_grad_norm(self.downsample_layers))
            self.log('Grad Norm/Attention Layers', calculate_grad_norm(self.attention_layers))
            self.log('Grad Norm/Upsample Layers', calculate_grad_norm(self.upsample_layers))
    # def configure_optimizers(self):
    #     return optim.Adam(self.parameters(), lr=1e-4)
    
@gin.configurable
class UNetPitchConditioned(UNetBase):
    def __init__(self,
                 inp_dim, 
                 time_dim, 
                 f0_dim,
                 features, 
                 strides, 
                 kernel_size, 
                 audio_seq_len, 
                 project_dim=None,
                 dropout=0.0,
                 nonlinearity=None,
                 norm=True,
                 num_convs=2,
                 num_attns=2,
                 num_heads=8,
                 log_samples_every=10, 
                 log_wandb_samples_every=10,
                 ckpt=None,
                 val_data=None,
                 qt=None,
                 singer_conditioning=False,
                 singer_dim=128,
                 singer_vocab=56,
                 sr = 44100,
                 cfg = False,
                 f0_mask = 0,
                 cond_drop_prob = 0.0,
                 groups = None,
                 nfft = None,
                 loss_w_padding = False,
                 log_grad_norms_every=10
                 ):
        super(UNetPitchConditioned, self).__init__()
        self.inp_dim = inp_dim
        self.time_dim = time_dim
        self.features = features
        self.strides = strides
        self.kernel_size = kernel_size
        self.seq_len = audio_seq_len
        self.log_samples_every = log_samples_every
        self.log_wandb_samples_every = log_wandb_samples_every
        self.ckpt = ckpt
        self.qt = qt
        self.singer_conditioning = singer_conditioning
        self.sr = sr    # used for logging audio to wandb
        self.cond_drop_prob = cond_drop_prob
        self.f0_masked_token = f0_mask 
        self.cfg = cfg
        self.strides_prod = np.prod(strides)
        self.loss_w_padding = loss_w_padding
        self.log_grad_norms_every = log_grad_norms_every

        conditioning_dim = time_dim
        if singer_conditioning:
            conditioning_dim += singer_dim

        if project_dim is None:
            project_dim = features[0]
        self.initial_projection = nn.Conv1d(inp_dim, project_dim, kernel_size=1)
        self.time_positional_encoding = PositionalEncoding(time_dim)
        self.f0_positional_encoding = PositionalEncoding(f0_dim)

        if singer_conditioning:
            self.singer_embedding = nn.Embedding(singer_vocab + 1*self.cfg, singer_dim) # if cfg, add 1 to the singer vocabulary
            self.singer_masked_token = singer_vocab
        else:
            self.singer_embedding = None

        features = [project_dim] + features
        f0_features = features.copy()
        f0_features[0] = f0_dim # first layer should be the f0 dimension
        strides = [None] + strides

        self.downsample_layers = nn.ModuleList([
            ResNetBlock(features[ind-1] + conditioning_dim, 
                        features[ind], 
                        kernel_size=kernel_size, 
                        stride=strides[ind],
                        dropout=dropout,
                        nonlinearity=nonlinearity,
                        norm=norm,
                        num_convs=num_convs,
                        ) for ind in range(1, len(features))
        ])
        
        self.f0_conv_layers = nn.ModuleList([
            nn.Conv1d(
                f0_dim,
                f0_dim,
                kernel_size=2 * strides[ind],
                stride=strides[ind],
                padding=strides[ind]//2,
            ) for ind in range(1, len(features))
        ])

        self.attention_layers = AttentionLayers(
            dim = features[-1],
            heads = num_heads,
            depth = num_attns,
        )

        self.upsample_layers = nn.ModuleList([
            ResNetBlock((features[ind] * 2) + (conditioning_dim) + f0_dim, # input size defined by features + skip dimension + time dimension  
            features[ind-1], 
            kernel_size=kernel_size, 
            stride=strides[ind],
            dropout=dropout,
            nonlinearity=nonlinearity,
            norm=norm,
            num_convs=num_convs,
            up=True
            ) for ind in range(len(features) - 1, 0, -1)
        ])
        self.final_projection = nn.Conv1d(2*project_dim + f0_dim, inp_dim, kernel_size=1)
        # save 16 f0 values from to sample on
        if val_data is not None:
            val_ids = np.random.choice(len(val_data), 16)
            val_samples = [val_data[i] for i in val_ids]
            self.val_f0 = torch.stack([v[1] for v in val_samples], 0).to(self.device)
            if self.singer_conditioning:
                self.val_singer = torch.tensor([v[2] for v in val_samples]).long().to(self.device)
            else:
                self.val_singer = None
            val_audio = torch.stack([v[0] for v in val_samples], 0).to(self.device)
            if self.ckpt is not None:
                # log the f0 and audio to wandb
                os.makedirs(os.path.join(self.ckpt, 'samples'), exist_ok=True)
                concat_audios = []
                beep = torch.sin(2 * torch.pi * 220 * torch.arange(0, 0.1 * self.sr) / self.sr).to(val_audio)
                recon_audios = p2a.normalized_mels_to_audio(val_audio, qt=self.qt)
                fig, axs = plt.subplots(4, 4, figsize=(16, 16))
                for i in range(4):
                    for j in range(4):
                        axs[i, j].plot(self.val_f0[i*4+j].squeeze())
                        if self.singer_conditioning:
                            axs[i, j].set_title(f'Singer {self.val_singer[i*4+j].item()}')
                        concat_audios.append(torch.cat((recon_audios[i*4+j].squeeze(), beep)))
                concat_audios = torch.cat(concat_audios, dim=-1).reshape(1, -1).to('cpu')
                output_file = os.path.join(self.ckpt, 'samples', f'gt_samples.wav')
                torchaudio.save(output_file, concat_audios, self.sr)
                
                try:
                    wandb.log({"sample f0 input": [wandb.Image(fig, caption="f0 conditioning on samples")]})
                    wandb.log({
                    "sample audio ground truth": [wandb.Audio(output_file, self.sr, caption="Samples")]})
                except:
                    pass
            
                fig.savefig(os.path.join(self.ckpt, 'samples', 'f0_inputs.png'))

    def pad_to(self, x, strides):
        # modified from: https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
        l = x.shape[-1]

        if l % strides > 0:
            new_l = l + strides - l % strides
        else:
            new_l = l

        ll, ul = int((new_l-l) / 2), int(new_l-l) - int((new_l-l) / 2)
        pads = (ll, ul)

        out = F.pad(x, pads, "reflect").to(x)

        return out, pads

    def unpad(self, x, pad):
        # modified from: https://stackoverflow.com/questions/66028743/how-to-handle-odd-resolutions-in-unet-architecture-pytorch
        if pad[0]+pad[1] > 0:
            x = x[:,:,pad[0]:-pad[1]]
        return x

    def forward(self, x, time, f0, singer, drop_tokens=True, drop_all=False):
        # INITIAL PROJECTION
        x = self.initial_projection(x)

        bs = x.shape[0]
        if self.cfg:
            # pdb.set_trace()
            if drop_all:
                prob_keep_mask_pitch = torch.zeros((bs)).unsqueeze(1).repeat(1, f0.shape[1]).to(self.device).bool()
                prob_keep_mask_singer = torch.zeros((bs)).to(self.device).bool()
            elif drop_tokens:
                prob_keep_mask_pitch = prob_mask_like((bs), 1. - self.cond_drop_prob, device = self.device).unsqueeze(1).repeat(1, f0.shape[1])
                prob_keep_mask_singer = prob_mask_like((bs), 1. - self.cond_drop_prob, device = self.device)
            else:
                prob_keep_mask_pitch = torch.ones((bs)).unsqueeze(1).repeat(1, f0.shape[1]).to(self.device).bool()
                prob_keep_mask_singer = torch.ones((bs)).to(self.device).bool()
            f0 = torch.where(prob_keep_mask_pitch, f0, torch.empty((f0.shape[0], f0.shape[1])).fill_(self.f0_masked_token).to(self.device).long())
            if self.singer_conditioning:
                singer = torch.where(prob_keep_mask_singer, singer, torch.empty((bs)).fill_(self.singer_masked_token).to(self.device).long())

        # TIME and F0 CONDITIONING
        conditions = [self.time_positional_encoding(time)]
        if self.singer_conditioning:
            conditions.append(self.singer_embedding(singer))
        f0 = self.f0_positional_encoding(f0)

        def _concat_condition(x, condition):
            condition = condition.unsqueeze(2).expand(-1, -1, x.shape[-1])
            x = torch.cat([x, condition], -2)
            return x

        skips = []

        # DOWNSAMPLING
        # pdb.set_trace()
        for ind, downsample_layer in enumerate(self.downsample_layers):
            # print(f'Down sample layer {ind}')
            # pdb.set_trace()
            skips.append(torch.cat([x, f0], -2))
            for cond in conditions:
                x = _concat_condition(x, cond)
            # print(x.shape, time.shape, f0.shape, skips[-1].shape)
            x = downsample_layer(x)
            f0 = self.f0_conv_layers[ind](f0)
        skips.append(torch.cat([x, f0], -2))
        # BOTTLENECK ATTENTION
        x = torch.permute(x, (0, 2, 1))
        x = self.attention_layers(x)
        x = torch.permute(x, (0, 2, 1))
        # print(x.shape, time.shape, f0.shape, skips[-1].shape)
        # pdb.set_trace()
        # UPSAMPLING
        for ind, upsample_layer in enumerate(self.upsample_layers):
            # print(f'Up sample layer {ind}')
            for cond in conditions:
                x = _concat_condition(x, cond)
            x = torch.cat([x, skips.pop(-1)], 1)
            # print(x.shape, time.shape, f0.shape)
            x = upsample_layer(x)
        x = torch.cat([x, skips.pop(-1)], 1)

        # FINAL PROJECTION
        x = self.final_projection(x)
        return x

    def loss(self, x, f0, singer, drop_tokens):
        # pdb.set_trace()
        padded_x, padding = self.pad_to(x, self.strides_prod)
        padded_f0, _ = self.pad_to(f0, self.strides_prod)
        t = torch.rand((padded_x.shape[0],)).to(padded_x)
        noise = torch.normal(0, 1, padded_x.shape).to(padded_x)
        # print(t.device, noise.device, x.device)
        x_t = t[:, None, None] * padded_x + (1 - t[:, None, None]) * noise
        # print(t.device, noise.device, x_t.device, x.device)
        padded_y = self.forward(x_t, t, padded_f0, singer, drop_tokens)
        unpadded_y = self.unpad(padded_y, padding)

        if self.loss_w_padding:
            target = padded_x - noise
            return torch.mean((padded_y - target) ** 2)
        else:
            target = x - self.unpad(noise, padding) # x1 - x0
            return torch.mean((unpadded_y - target) ** 2)

    def training_step(self, batch, batch_idx):
        # print('\n', batch_idx, batch.shape)
        x, f0, singer = batch
        x = x.to(self.device)
        f0 = f0.to(self.device)
        singer = singer.reshape(-1).long().to(self.device) if self.singer_conditioning else None
        loss = self.loss(x, f0, singer, drop_tokens=True)
        self.log('train_loss', loss, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        # pdb.set_trace()
        x, f0, singer = batch
        x = x.to(self.device)
        f0 = f0.to(self.device)
        singer = singer.reshape(-1).long().to(self.device) if self.singer_conditioning else None
        loss = self.loss(x, f0, singer, drop_tokens=False)
        self.log('val_loss', loss, batch_size=x.shape[0])
        return loss
    
    def sample_fn(self, f0, singer, batch_size: int, num_steps: int):
        # CREATE INITIAL NOISE
        noise = torch.normal(mean=0, std=1, size=(batch_size, self.inp_dim, self.seq_len)).to(self.device)
        padded_noise, padding = self.pad_to(noise, self.strides_prod)
        t_array = torch.ones((batch_size,)).to(self.device)
        f0 = f0.to(self.device)
        padded_f0, _ = self.pad_to(f0, self.strides_prod)
        singer = singer.to(self.device)
        with torch.no_grad():
            # SAMPLE FROM MODEL
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                t_tensor = torch.tensor(t)
                padded_noise = padded_noise + 1 / num_steps * self.forward(padded_noise, t_tensor * t_array, padded_f0, singer, drop_tokens=False)
        noise = self.unpad(padded_noise, padding)
        return noise

    def sample_cfg(self, batch_size: int, num_steps: int, f0=None, singer=[4, 25, 45, 32], strength=1, invert_audio_fn=None, log_interim_samples=False):
        # CREATE INITIAL NOISE
        noise = torch.normal(mean=0, std=1, size=(batch_size, self.inp_dim, self.seq_len)).to(self.device)
        padded_noise, padding = self.pad_to(noise, self.strides_prod)
        t_array = torch.ones((batch_size,)).to(self.device)
        if f0 is None:
            val_idx = np.random.choice(len(self.val_dataloader), batch_size)
            val_samples = [self.val_dataloader[i][1] for i in val_idx]
            f0 = torch.stack([sample for sample in val_samples]).to(self.device)
        else:
            assert len(f0) == batch_size
            f0 = f0.to(self.device)
            singer = singer.to(self.device)
            # f0 = torch.tensor(f0).to(self.device)
        # singer = torch.Tensor(np.choice(singer, batch_size, replace=True)).to(self.device)
        padded_f0, _ = self.pad_to(f0, self.strides_prod)
        with torch.no_grad():
            # SAMPLE FROM MODEL
            for t in np.linspace(0, 1, num_steps + 1)[:-1]:
                t_tensor = torch.tensor(t)
                    
                unconditioned_logits = self.forward(padded_noise, t_tensor * t_array, padded_f0, singer, drop_tokens=False, drop_all=True)
                conditioned_logits = self.forward(padded_noise, t_tensor * t_array, padded_f0, singer, drop_tokens=False, drop_all=False)
                total_logits = strength * conditioned_logits + (1 - strength) * unconditioned_logits
                padded_noise = padded_noise + 1 / num_steps * total_logits
            
            noise = self.unpad(padded_noise, padding)
        return noise, f0, singer

    def on_validation_epoch_end(self) -> None:
        with torch.no_grad():
            # pdb.set_trace()
            if self.current_epoch % self.log_samples_every == 0:
                samples = self.sample_fn(self.val_f0, self.val_singer, 16, 100)
                if self.ckpt is not None:
                    audio = p2a.normalized_mels_to_audio(samples, qt=self.qt)
                    beep = torch.sin(2 * torch.pi * 220 * torch.arange(0, 0.1 * self.sr) / self.sr).to(audio)
                    concat_audio = []
                    for sample in audio:
                        concat_audio.append(torch.cat([sample, beep]))
                    concat_audio = torch.cat(concat_audio, dim=-1).reshape(1, -1).to('cpu')
                    output_file = os.path.join(self.ckpt, 'samples', f'samples_{self.current_epoch}.wav')
                    torchaudio.save(output_file, concat_audio, self.sr)
                if self.current_epoch % self.log_wandb_samples_every == 0:
                    if self.logger:
                        wandb.log({
                            "samples": [wandb.Audio(output_file, self.sr, caption="Samples")]},
                            step = self.global_step)
    def on_before_optimizer_step(self, optimizer, *_):
        def calculate_grad_norm(module_list, norm_type=2):
            total_norm = 0
            if isinstance(module_list, nn.Module):
                module_list = [module_list]
            for module in module_list:
                for name, param in module.named_parameters():
                    if param.requires_grad:
                        param_norm = torch.norm(param.grad.detach(), p=norm_type)
                        total_norm += param_norm**2
            # pdb.set_trace()
            total_norm = torch.sqrt(total_norm)
            return total_norm
    
        if self.log_grad_norms_every is not None and self.global_step % self.log_grad_norms_every == 0:
            self.log('Grad Norm/Downsample Layers', calculate_grad_norm(self.downsample_layers))
            self.log('Grad Norm/Attention Layers', calculate_grad_norm(self.attention_layers))
            self.log('Grad Norm/Upsample Layers', calculate_grad_norm(self.upsample_layers))

    # @gin.configurable
    # def configure_optimizers(self, optimizer_cls: Callable[[], torch.optim.Optimizer],
    # scheduler_cls: Callable[[],
    #                         torch.optim.lr_scheduler._LRScheduler]):
    #     # pdb.set_trace()
    #     optimizer = optimizer_cls(self.parameters())
    #     scheduler = scheduler_cls(optimizer)

    #     return [optimizer], [{'scheduler': scheduler, 'interval': 'step'}]
