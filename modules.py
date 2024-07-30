import torch.nn.functional as F
from layers import *
from PIL import Image
from torchvision.transforms.functional import resize
from einops import rearrange

class SPSA_Attention(nn.Module):
    def __init__(self, dim, num_heads,is_material_mask, is_spec, bias):
        super(SPSA_Attention, self).__init__()
        self.num_heads = num_heads
        self.is_material_mask = is_material_mask
        self.is_spec = is_spec
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        if self.is_spec:
            self.project_out1_x = nn.Conv2d(dim//2, dim//2,  kernel_size=3, stride=1, padding=1, bias=bias)
            self.project_out1_spec = nn.Conv2d(dim//2, dim//2,  kernel_size=3, stride=1, padding=1, bias=bias)
            self.project_out2_x = nn.Conv2d(dim, dim//2,  kernel_size=3, stride=1, padding=1, bias=bias)
            self.project_out2_spec = nn.Conv2d(dim, dim//2,  kernel_size=3, stride=1, padding=1, bias=bias)
        else:
            self.project_out1 = nn.Conv2d(dim, dim,  kernel_size=3, stride=1, padding=1, bias=bias)
            self.project_out2 = nn.Conv2d(dim, dim,  kernel_size=3, stride=1, padding=1, bias=bias)

        # ===========================mask condition===========================
        self.ResBlock_SFTk = ResBlock_SFT(input_channel = dim,input_mask_dim=1)
        self.ResBlock_SFTq = ResBlock_SFT(input_channel = dim,input_mask_dim=1)
        self.out_sft1 = ResBlock_SFT(input_channel = dim,input_mask_dim=1)
        self.out_sft2 = ResBlock_SFT(input_channel = dim,input_mask_dim=1)
        self.out_sft3 = ResBlock_SFT(input_channel = dim//2,input_mask_dim=1)
        self.out_sft4 = ResBlock_SFT(input_channel = dim//2,input_mask_dim=1)

    def forward(self, x_in,spec, material_mask):
        x = torch.cat([spec,x_in],dim=1)
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)

        #material semantic prior
        if self.is_material_mask:
            q = self.ResBlock_SFTk(q,material_mask)
            k = self.ResBlock_SFTq(k,material_mask)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        x_out = self.project_out2_x(out) + self.project_out1_x(x_in)
        spec_out = self.project_out2_spec(out) + self.project_out1_spec(spec)
        return x_out,spec_out

class SFTLayer(nn.Module):
    def __init__(self,dim,input_mask_dim):
        super(SFTLayer, self).__init__()
        self.SFT_scale_conv0 = nn.Conv2d(input_mask_dim, dim//2, kernel_size=1, stride=1, padding=0, bias=True) #nn.Conv2d(32, 32, 1)
        self.SFT_scale_conv1 = nn.Conv2d(dim//2, dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.SFT_shift_conv0 = nn.Conv2d(input_mask_dim, dim//2, kernel_size=1, stride=1, padding=0, bias=True)
        self.SFT_shift_conv1 = nn.Conv2d(dim//2, dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x,seg):
        bt, c, h, w = x.shape
        seg = resize(seg, (h, w), Image.BILINEAR)
        scale = self.SFT_scale_conv1(F.leaky_relu(self.SFT_scale_conv0(seg), 0.1, inplace=True))
        shift = self.SFT_shift_conv1(F.leaky_relu(self.SFT_shift_conv0(seg), 0.1, inplace=True))
        return x * (scale + 1) + shift
class ResBlock_SFT(nn.Module):
    def __init__(self,input_channel,input_mask_dim):
        super(ResBlock_SFT, self).__init__()
        self.sft0 = SFTLayer(dim = input_channel,input_mask_dim = input_mask_dim)
        self.conv0 = nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.sft1 = SFTLayer(dim = input_channel,input_mask_dim = input_mask_dim)
        self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x,seg):
        # x[0]: fea; x[1]: cond
        fea = self.sft0(x,seg)
        fea = F.relu(self.conv0(fea), inplace=True)
        fea = self.sft1(fea, seg)
        fea = self.conv1(fea)
        return x + fea


class SegExtract(nn.Module):
    def __init__(self, params, c_in=1):
        super(SegExtract, self).__init__()
        self.params = params
        self.relu = nn.ReLU()

        self.splat1 = nn.Conv2d(c_in, 8, kernel_size=3, stride=1, padding=1, bias=True)#conv_layer(c_in, 8, kernel_size=3, stride=1, padding=1, batch_norm=False)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=(1,1))#
        self.splat2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True)#conv_layer(8, 16, kernel_size=3, stride=1, padding=1, batch_norm=params['batch_norm'])
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  #
        #
        self.splat1_up = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=True)#conv_layer(16, 8, kernel_size=3, stride=1, padding=1, batch_norm=params['batch_norm'])
        self.splat2_up = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)#conv_layer(8, 1, kernel_size=3, stride=1, padding=1, batch_norm=params['batch_norm'])
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_in):
        x_in = resize(x_in, (16, 16), Image.BILINEAR)

        x1 = self.splat1(x_in)
        x1 = self.maxpool1(x1)
        x1 = self.splat2(x1)
        x_low1 = self.maxpool2(x1)

        x1 = self.splat1_up(x_low1)
        x1 = F.interpolate(x1, size=(8,8), mode='bilinear')
        x1 = self.splat2_up(x1)
        a = F.interpolate(x1, size=(16,16), mode='bilinear')#self.upsamp2(x1)

        out = 1.0+x_in*(1+a)
        return out

class BrightnessAdaptation(nn.Module):
    def __init__(self, params, c_in=1):
        super(BrightnessAdaptation, self).__init__()
        self.params = params
        self.relu = nn.ReLU()

        self.splat1 = nn.Conv2d(c_in, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.splat1_2 = nn.Conv2d(c_in, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3,3),stride=(2,2),padding=(1,1))#
        self.splat2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.splat2_2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1, bias=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  #

        self.splat1_up = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.splat1_up2 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.upsamp1 = torch.nn.Upsample(size=(params['output_res'][0]//2,params['output_res'][1]//2), mode='bilinear')
        self.splat2_up = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.splat2_up2 = nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.upsamp2 = torch.nn.Upsample(size=(params['output_res'][0],params['output_res'][1]), mode='bilinear')
        self.sigmoid = nn.Sigmoid()
    def forward(self, x_in,fullres):
        x1 = self.splat1(x_in)
        x1 = self.maxpool1(x1)
        x1 = self.splat2(x1)
        x_low1 = self.maxpool2(x1)

        x1 = self.splat1_up(x_low1)
        x1 = F.interpolate(x1, size=(fullres.size()[-2:][0]//2,fullres.size()[-2:][1]//2), mode='bilinear')
        x1 = self.splat2_up(x1)
        a = F.interpolate(x1, size=(fullres.size()[-2:][0],fullres.size()[-2:][1]), mode='bilinear')

        x2 = self.splat1_2(x_in)
        x2 = self.maxpool1(x2)
        x2 = self.splat2_2(x2)
        x_low2 = self.maxpool2(x2)

        x2 = self.splat1_up2(x_low2)
        x2 = F.interpolate(x2, size=(fullres.size()[-2:][0]//2,fullres.size()[-2:][1]//2), mode='bilinear')
        x2 = self.splat2_up2(x2)
        b = F.interpolate(x2, size=(fullres.size()[-2:][0],fullres.size()[-2:][1]), mode='bilinear')
        out = x_in*(1+a)+b
        out = torch.clamp(out, 0.01, 1)
        return out
