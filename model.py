import torch
import torch.nn as nn
from torch.autograd import Function
import torchvision.models as models
import ctlib
import swin_transformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class bfan_ed(nn.Module):
    def __init__(self, options):
        super().__init__()
        dets = int(options[1])
        dDet = options[5]
        s2r = options[8]
        d2r = options[9]
        self.virdet = dDet * s2r / (s2r + d2r)
        filter = torch.empty(2 * dets - 1)
        pi = torch.pi
        for i in range(filter.size(0)):
            x = i - dets + 1
            if abs(x) % 2 == 1:
                filter[i] = -1 / (pi * pi * x * x * self.virdet * self.virdet)
            elif x == 0:
                filter[i] = 1 / (4 * self.virdet * self.virdet)
            else:
                filter[i] = 0
        self.w = torch.arange((-dets / 2 + 0.5), dets / 2) * self.virdet
        self.w = s2r / torch.sqrt(s2r ** 2 + self.w ** 2)
        self.w = self.w.view(1,1,1,-1).cuda()
        self.filter = filter.view(1,1,1,-1).cuda()
        self.options = nn.Parameter(options, requires_grad=False)
        self.dets = dets
        self.coef = pi / options[0]

    def forward(self, projection):
        p = projection * self.virdet * self.w
        p = torch.nn.functional.conv2d(p, self.filter, padding=(0,self.dets-1))
        recon = bprj_fun.apply(p, self.options)
        recon = recon * self.coef
        return recon

class bprj_fun(Function):
    @staticmethod
    def forward(self, proj, options):
        self.save_for_backward(options)
        return ctlib.backprojection(proj, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        grad_input = ctlib.backprojection_t(grad_output.contiguous(), options)
        return grad_input, None

class prj_fun(Function):
    @staticmethod
    def forward(self, img, options):
        self.save_for_backward(options)
        return ctlib.projection(img, options)

    @staticmethod
    def backward(self, grad_output):
        options = self.saved_tensors[0]
        grad_input = ctlib.projection_t(grad_output.contiguous(), options)
        return grad_input, None

class prj_module(nn.Module):
    def __init__(self, options):
        super(prj_module, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(1).squeeze())
        self.options = nn.Parameter(options, requires_grad=False)
        self.fbp = bfan_ed(options)
        self.weight.data.zero_()
        
    def forward(self, input_data, proj):
        p_tmp = prj_fun.apply(input_data, self.options)
        y_error = proj - p_tmp
        x_error = self.fbp(y_error)
        out = self.weight * x_error + input_data
        return out

class ConvBlock(nn.Module):
    def __init__(self, dim, first=False, last=False) -> None:
        super().__init__()
        if first:
            self.conv1 = nn.Conv2d(1, dim, kernel_size=5, padding=2)
        else:
            self.conv1 = nn.Conv2d(1, dim // 2, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(dim, 1, kernel_size=5, padding=2)        
        if last:
            self.trans_embed = None
        else:
            self.trans_embed = nn.Conv2d(dim, dim // 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y):        
        x = self.relu(self.conv1(x))
        if not y is None:
            x = torch.cat((x, y), dim=1)
        x = self.relu(self.conv2(x))
        out = self.conv3(x)
        if self.trans_embed is None:
            z = None
        else:
            z = self.trans_embed(x)
        return out, z

class IterBlock(nn.Module):
    def __init__(self, options, idx, last=False):
        super(IterBlock, self).__init__()
        self.block1 = prj_module(options)
        first = True if idx == 0 else False
        if (idx % 2 == 0):
            self.block2 = ConvBlock(96, first=first, last=last)
        else:
            self.block2 = Transformer(first=first, last=last)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input_data, proj, z):
        tmp1 = self.block1(input_data, proj)
        tmp2, z_out = self.block2(input_data, z)
        output = tmp1 + tmp2
        output = self.relu(output)
        return output, z_out

class regformer(nn.Module):
    def __init__(self, block_num, **kwargs):
        super(regformer, self).__init__()
        views = kwargs['views']
        dets = kwargs['dets']
        width = kwargs['width']
        height = kwargs['height']
        dImg = kwargs['dImg']
        dDet = kwargs['dDet']
        dAng = kwargs['dAng']
        s2r = kwargs['s2r']
        d2r = kwargs['d2r']
        binshift = kwargs['binshift']
        options = torch.Tensor([views, dets, width, height, dImg, dDet, 0, dAng, s2r, d2r, binshift, 0])
        self.model = nn.ModuleList([IterBlock(options, i, last=True if i == block_num - 1 else False) for i in range(block_num)])
    
    def forward(self, input_data, proj):
        x = input_data
        z = None
        for index, module in enumerate(self.model):
            x, z = module(x, proj, z)
        return x

class Transformer(nn.Module):
    def __init__(self, img_size=256, embed_dim=96, depths=[2], num_heads=[3],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, first=False, last = False, **kwargs):
        super(Transformer, self).__init__()
        self.num_layers = len(depths)
        self.mlp_ratio = mlp_ratio
        self.patch_norm = patch_norm
        if not first:
            self.patch_embed = nn.Conv2d(1, embed_dim // 2, 3, 1, 1)
        else:
            self.patch_embed = nn.Conv2d(1, embed_dim, 3, 1, 1)
        self.embed_reverse = nn.Conv2d(embed_dim, 1, 3, 1, 1)
        if not last:
            self.trans_embed = nn.Conv2d(embed_dim, embed_dim // 2, 3, 1, 1)
        else:
            self.trans_embed = None
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = swin_transformer.BasicLayer(dim=embed_dim,
                               input_resolution=(img_size, img_size),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 0.01)

    def forward(self, x, y):
        B, C, H, W = x.shape

        x = self.patch_embed(x)
        if not y is None:
            x = torch.cat((x,y), dim=1)
        x = x.flatten(2).transpose(1, 2)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x).transpose(1,2).view(B, -1, H, W)
        out = self.embed_reverse(x)
        if self.trans_embed is None:
            z = None
        else:
            z = self.trans_embed(x)
        return out, z

