import numpy as np
from layers import *
from modules import SPSA_Attention, BrightnessAdaptation, SegExtract

class FeatureExtract(nn.Module):
    def __init__(self, params, c_in = 3):
        super(FeatureExtract, self).__init__()
        self.params = params
        self.relu = nn.ReLU()
        # ===========================attention===========================
        if self.params['spec']:
            self.attn1 = SPSA_Attention(dim=8*2, num_heads=1, is_material_mask = self.params['material_mask'], is_spec = self.params['spec'], bias=False)
            self.attn2 = SPSA_Attention(dim=16*2, num_heads=1,is_material_mask = self.params['material_mask'], is_spec = self.params['spec'], bias=False)
            self.attn3 = SPSA_Attention(dim=32*2, num_heads=1,is_material_mask = self.params['material_mask'], is_spec = self.params['spec'], bias=False)
            self.attn4 = SPSA_Attention(dim=64*2, num_heads=1,is_material_mask = self.params['material_mask'], is_spec = self.params['spec'], bias=False)
        else:
            self.attn1 = SPSA_Attention(dim=8, num_heads=1, is_material_mask = self.params['material_mask'], is_spec = self.params['spec'], bias=False)
            self.attn2 = SPSA_Attention(dim=16, num_heads=1,is_material_mask = self.params['material_mask'], is_spec = self.params['spec'], bias=False)
            self.attn3 = SPSA_Attention(dim=32, num_heads=1,is_material_mask = self.params['material_mask'], is_spec = self.params['spec'], bias=False)
            self.attn4 = SPSA_Attention(dim=64, num_heads=1,is_material_mask = self.params['material_mask'], is_spec = self.params['spec'], bias=False)
        # ===========================Fusion===========================
        self.fusion1 = conv_layer(16, 8,  kernel_size=3, stride=1, padding=1, batch_norm=params['batch_norm'])
        self.fusion2 = conv_layer(32, 16,  kernel_size=3, stride=1, padding=1, batch_norm=params['batch_norm'])
        self.fusion3 = conv_layer(64, 32,  kernel_size=3, stride=1, padding=1, batch_norm=params['batch_norm'])
        self.fusion4 = conv_layer(128, 64,  kernel_size=3, stride=1, padding=1, batch_norm=params['batch_norm'])
        # ===========================Splat===========================
        self.splat1 = conv_layer(c_in, 8,  kernel_size=3, stride=2, padding=1, batch_norm=False)
        self.splat2 = conv_layer(8,    16, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.splat3 = conv_layer(16,   32, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.splat4 = conv_layer(32,   64, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])

        self.splat1_spec = conv_layer(10,    8, kernel_size=3, stride=2, padding=1, batch_norm=False) #12.18
        self.splat2_spec = conv_layer(8,    16, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.splat3_spec = conv_layer(16,   32, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.splat4_spec = conv_layer(32,   64, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        # ===========================Global mine===========================
        # Conv until 4x4
        self.global1 = conv_layer(64, 128, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.global2 = conv_layer(128, 256, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.global3 = conv_layer(256, 128, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.global4 = conv_layer(128, 64, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])
        self.global5 = conv_layer(64, 64, kernel_size=3, stride=2, padding=1, batch_norm=params['batch_norm'])

        # ===========================Local===========================
        self.local1 = conv_layer(64, 64, kernel_size=3, padding=1, batch_norm=params['batch_norm'])
        self.local2 = conv_layer(64, 64, kernel_size=3, padding=1, bias=False, activation=None)

        # ===========================predicton===========================
        self.pred = conv_layer(64, 96, kernel_size=1, activation=None) # 64 -> 96

    def forward(self, x, spec, material_mask):
        N = x.shape[0]
        # ===========================Splat===========================
        x = self.splat1(x) # N, C=8,  H=128, W=128
        if self.params['spec']:
            spec = self.splat1_spec(spec) # N, C=8,  H=128, W=128
            x,spec = self.attn1(x, spec, material_mask)

        x = self.splat2(x) # N, C=16, H=64,  W=64
        if self.params['spec']:
            spec = self.splat2_spec(spec) # N, C=8,  H=128, W=128
            x,spec = self.attn2(x, spec, material_mask)

        x = self.splat3(x) # N, C=32, H=32,  W=32
        if self.params['spec']:
            spec = self.splat3_spec(spec) # N, C=8,  H=128, W=128
            x,spec = self.attn3(x, spec, material_mask)

        x = self.splat4(x) # N, C=64, H=16,  W=16
        if self.params['spec']:
            spec = self.splat4_spec(spec) # N, C=8,  H=128, W=128
            x,spec = self.attn4(x, spec, material_mask)

        splat_out = x # N, C=64, H=16,  W=16
        # ===========================Global mine===========================
        # convs
        x = self.global1(x)
        x = self.global2(x)
        # flatten
        x = self.global3(x)
        x = self.global4(x)
        x = self.global5(x)
        global_out = x.squeeze(2).squeeze(2)
        # ===========================Local===========================
        x = splat_out
        x = self.local1(x)
        x = self.local2(x)
        local_out = x
        # ===========================Fusion===========================
        global_out = global_out[:, :, None, None] # N, 64， 1， 1
        fusion = self.relu(local_out + global_out) # N, C=64, H=16, W=16
        # ===========================Prediction===========================
        x = self.pred(fusion) # N, C=96, H=16, W=16
        x = x.view(N, 12, 8, 16, 16)#16, 16) # N, C=12, D=8, H=16, W=16
        return x

class Coefficients(nn.Module):
    def __init__(self, params, c_in=3):
        super(Coefficients, self).__init__()
        self.params = params
        self.relu = nn.ReLU()
        # ===========================FeatureExtract===========================
        self.FeatureExtract0 = FeatureExtract(params,c_in=3)
        self.FeatureExtract1 = FeatureExtract(params,c_in=3)
        self.FeatureExtract2 = FeatureExtract(params,c_in=3)
        self.FeatureExtract3 = FeatureExtract(params,c_in=3)
        self.FeatureExtract4 = FeatureExtract(params,c_in=3)
        self.FeatureExtract5 = FeatureExtract(params,c_in=3)
        self.SegExtract0 = SegExtract(params)
        self.SegExtract1 = SegExtract(params)
        self.SegExtract2 = SegExtract(params)
        self.SegExtract3 = SegExtract(params)
        self.SegExtract4 = SegExtract(params)
        self.SegExtract5 = SegExtract(params)

    def forward(self, x, spec, material_mask):
        #FeatureExtract
        if self.params['material_mask']:
            x0 = self.FeatureExtract0(x,spec, material_mask[:,0,:,:].unsqueeze(1))
            x1 = self.FeatureExtract1(x,spec, material_mask[:,1,:,:].unsqueeze(1))
            x2 = self.FeatureExtract2(x,spec, material_mask[:,2,:,:].unsqueeze(1))
            x3 = self.FeatureExtract3(x,spec, material_mask[:,3,:,:].unsqueeze(1))
            x4 = self.FeatureExtract4(x,spec, material_mask[:,4,:,:].unsqueeze(1))
            x5 = self.FeatureExtract5(x,spec, material_mask[:,5,:,:].unsqueeze(1))
            x0_seg = self.SegExtract0(material_mask[:,0,:,:].unsqueeze(1)).unsqueeze(1)
            x1_seg = self.SegExtract1(material_mask[:,1,:,:].unsqueeze(1)).unsqueeze(1)
            x2_seg = self.SegExtract2(material_mask[:,2,:,:].unsqueeze(1)).unsqueeze(1)
            x3_seg = self.SegExtract3(material_mask[:,3,:,:].unsqueeze(1)).unsqueeze(1)
            x4_seg = self.SegExtract4(material_mask[:,4,:,:].unsqueeze(1)).unsqueeze(1)
            x5_seg = self.SegExtract5(material_mask[:,5,:,:].unsqueeze(1)).unsqueeze(1)
            x = x0*x0_seg + x1*x1_seg + x2*x2_seg + x3*x3_seg + x4*x4_seg + x5*x5_seg
        else:
            x = self.FeatureExtract0(x,spec, material_mask[:,0,:,:].unsqueeze(1))
        return x


class Guide(nn.Module):
    def __init__(self, params, c_in=3):
        super(Guide, self).__init__()
        self.params = params
        # Number of relus/control points for the curve
        self.nrelus = 16
        self.c_in = c_in
        self.M = nn.Parameter(torch.eye(c_in, dtype=torch.float32) + torch.randn(1, dtype=torch.float32) * 1e-4) # (c_in, c_in)
        self.M_bias = nn.Parameter(torch.zeros(c_in, dtype=torch.float32)) # (c_in,)
        # The shifts/thresholds in x of relus
        thresholds = np.linspace(0, 1, self.nrelus, endpoint=False, dtype=np.float32) # (nrelus,)
        thresholds = torch.tensor(thresholds) # (nrelus,)
        thresholds = thresholds[None, None, None, :] # (1, 1, 1, nrelus)
        thresholds = thresholds.repeat(1, 1, c_in, 1) # (1, 1, c_in, nrelus)
        self.thresholds = nn.Parameter(thresholds) # (1, 1, c_in, nrelus)
        # The slopes of relus
        slopes = torch.zeros(1, 1, 1, c_in, self.nrelus, dtype=torch.float32) # (1, 1, 1, c_in, nrelus)
        slopes[:, :, :, :, 0] = 1.0
        self.slopes = nn.Parameter(slopes)

        self.relu = nn.ReLU()
        self.bias = nn.Parameter(torch.tensor(0, dtype=torch.float32))

    def forward(self, x,material_mask,nir):
        x = x.permute(0, 2, 3, 1) # N, H, W, C=3
        old_shape = x.shape # (N, H, W, C=3)

        x = torch.matmul(x.reshape(-1, self.c_in), self.M) # N*H*W, C=3
        x = x + self.M_bias
        x = x.reshape(old_shape) # N, H, W, C=3
        x = x.unsqueeze(4) # N, H, W, C=3, 1
        x = torch.sum(self.slopes * self.relu(x - self.thresholds), dim=4) # N, H, W, C=3

        x = x.permute(0, 3, 1, 2) # N, C=3, H, W

        x = torch.sum(x, dim=1, keepdim=True) / self.c_in # N, C=1, H, W
        x = x + self.bias # N, C=1, H, W
        x = torch.clamp(x, 0, 1) # N, C=1, H, W
        return x


class JDMHDRnetModel(nn.Module):
    def __init__(self, params):
        super(JDMHDRnetModel, self).__init__()
        self.coefficients = Coefficients(params)
        self.BrightnessAdaptation1 = BrightnessAdaptation(params)
        self.BrightnessAdaptation2 = BrightnessAdaptation(params)
        self.BrightnessAdaptation3 = BrightnessAdaptation(params)
        self.guide = Guide(params)

    def forward(self, lowres, fullres,spec,material_mask,nir):
        #step1 Brightness Adaptation
        hue = self.BrightnessAdaptation1(nir,fullres)
        hue_out = self.BrightnessAdaptation2(nir,fullres)
        hue_spec = self.BrightnessAdaptation3(nir, fullres)
        hue_lowres = F.interpolate(hue, size=lowres.size()[-2:],mode='bilinear')
        hue_spec = F.interpolate(hue_spec, size=lowres.size()[-2:],mode='bilinear')
        fullres = fullres/hue
        lowres = lowres/hue_lowres
        spec = spec/hue_spec
        # step2 grid coefficient predict
        grid = self.coefficients(lowres,spec,material_mask)# N, C=12, D=8, H=16, W=16
        # step3 guide map
        guide = self.guide(fullres,material_mask,nir) # N, C=1, H, W
        #step4 slicing
        sliced = slicing(grid, guide)
        #step5 generate output
        output = apply(sliced, fullres)
        output = output * hue_out

        return output

