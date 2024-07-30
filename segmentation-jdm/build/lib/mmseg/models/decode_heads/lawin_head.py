import torch
import torch.nn.functional as F
from torch import nn
from mmcv.cnn import ConvModule, NonLocal2d
from mmseg.ops.wrappers import resize
from .decode_head import BaseDecodeHead
from einops import rearrange
from ..builder import HEADS


class MLP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.mlp = nn.Linear(in_channels, out_channels)

    def forward(self, x: torch.Tensor):
        x = x.flatten(start_dim=2).transpose(1, 2)
        x = self.mlp(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch Embedding: github.com/SwinTransformer/
    """

    def __init__(self,
                 proj_type='pool',
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj_type = proj_type
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if proj_type == 'conv':
            self.proj = nn.Conv2d(
                in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        elif proj_type == 'pool':
            self.proj = nn.ModuleList([
                nn.MaxPool2d(kernel_size=patch_size, stride=patch_size),
                nn.AvgPool2d(kernel_size=patch_size, stride=patch_size)
            ])
        else:
            raise NotImplementedError(
                f'{proj_type} is not currently supported.')

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()
        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x,
                      (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        if self.proj_type == 'conv':
            x = self.proj(x)  # B C Wh Ww
        else:
            x = 0.5 * (self.proj[0](x) + self.proj[1](x))

        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)
        return x


class LawinAttn(NonLocal2d):

    def __init__(self, head: int, patch_size: int, **kwargs):
        super().__init__(**kwargs)
        self.head = head
        self.patch_size = patch_size
        size = patch_size**2
        # MLP Mixer
        self.position_mixing = nn.ModuleList(
            [nn.Linear(size, size) for _ in range(head)])

    def forward(self, query: torch.Tensor, context: torch.Tensor):
        # context indicates nearby patches of query object
        b, c, h, w = context.shape

        if self.head != 1:
            # MLP output for context input
            context = context.reshape(b, c, -1)
            out = []
            for i in range(self.head):
                cube = context[:, (c // self.head) * i:(c // self.head) *
                               (i + 1), :]
                out.append(self.position_mixing[i](cube))
            out = torch.cat(out, dim=1)
            context += out
            context = context.reshape(b, c, h, w)

        # this part is copied from base NonLocal2d forward function
        g_x = self.g(context).view(b, self.inter_channels, -1)
        g_x = rearrange(g_x, "n (h dim) b -> (n h) dim b", h=self.head)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = query.view(b, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            if self.sub_sample:
                phi_x = self.phi(context).view(b, self.in_channels, -1)
            else:
                phi_x = context.view(b, self.in_channels, -1)
        elif self.mode == "concatenation":
            theta_x = self.theta(query).view(b, self.inter_channels, -1, 1)
            phi_x = self.phi(context).view(b, self.inter_channels, 1, -1)
        else:
            theta_x = self.theta(query).view(b, self.inter_channels, -1)
            theta_x = rearrange(
                theta_x, "n (h dim) b -> (n h) dim b", h=self.head)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(context).view(b, self.inter_channels, -1)
            phi_x = rearrange(phi_x, "n (h dim) b -> (n h) dim b", h=self.head)

        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)

        y = torch.matmul(pairwise_weight, g_x)
        y = rearrange(y, "(b h) n dim -> b n (h dim)", h=self.head)
        y = y.permute(0, 2, 1).contiguous().reshape(b, self.inter_channels,
                                                    *query.size()[2:])

        output = query + self.conv_out(y)
        return output


@HEADS.register_module()
class LawinHead(BaseDecodeHead):

    def __init__(
        self,
        embed_dim: int = 768,
        reduction: int = 2,
        use_scale: bool = True,
        patch_size: int = 8,
        **kwargs,
    ):
        super(LawinHead, self).__init__(
            **kwargs,
            input_transform='multiple_select',
        )

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # Lawin Attention blocks
        self.em_8 = PatchEmbed(
            proj_type='pool',
            patch_size=8,
            in_chans=512,
            embed_dim=512,
            norm_layer=nn.LayerNorm,
        )
        self.em_4 = PatchEmbed(
            proj_type='pool',
            patch_size=4,
            in_chans=512,
            embed_dim=512,
            norm_layer=nn.LayerNorm,
        )
        self.em_2 = PatchEmbed(
            proj_type='pool',
            patch_size=2,
            in_chans=512,
            embed_dim=512,
            norm_layer=nn.LayerNorm,
        )

        self.lawin_8 = LawinAttn(
            in_channels=512,
            reduction=reduction,
            use_scale=use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode='embedded_gaussian',
            head=8**2,
            patch_size=patch_size,
        )
        self.lawin_4 = LawinAttn(
            in_channels=512,
            reduction=reduction,
            use_scale=use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode='embedded_gaussian',
            head=4**2,
            patch_size=patch_size,
        )
        self.lawin_2 = LawinAttn(
            in_channels=512,
            reduction=reduction,
            use_scale=use_scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            mode='embedded_gaussian',
            head=2**2,
            patch_size=patch_size,
        )
        self.lawin_none = ConvModule(
            in_channels=512,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )
        self.image_pooling = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(
                in_channels=512,
                out_channels=512,
                kernel_size=1,
                conv_cfg=self.conv_cfg,
                act_cfg=self.act_cfg,
            ),
        )

        # MLPs for input fusion
        self.mlp_1 = MLP(self.in_channels[0], out_channels=48)
        self.mlp_2 = MLP(self.in_channels[1], embed_dim)
        self.mlp_3 = MLP(self.in_channels[2], embed_dim)
        self.mlp_4 = MLP(self.in_channels[3], embed_dim)

        # Cat
        self.cat_1 = ConvModule(
            in_channels=embed_dim * 3,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )
        self.cat_2 = ConvModule(
            in_channels=512 * 5,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )
        self.cat_3 = ConvModule(
            in_channels=512 + 48,
            out_channels=512,
            kernel_size=1,
            norm_cfg=self.norm_cfg,
        )

    def _get_context(self, data: torch.Tensor, patch_size: int = 8):
        b, _, h, w = data.shape
        context = []
        for r in [8, 4, 2]:
            _context = F.unfold(
                data,
                kernel_size=patch_size * r,
                stride=patch_size,
                padding=int((r - 1) / 2 * patch_size),
            )
            _context = rearrange(
                _context,
                'b (c ph pw) (nh nw) -> (b nh nw) c ph pw',
                ph=patch_size * r,
                pw=patch_size * r,
                nh=h // patch_size,
                nw=w // patch_size,
            )
            context.append(getattr(self, f'em_{r}')(_context))
        return context

    def forward(self, x: torch.Tensor):
        # inputs = [x1, x2, x3, x4] from BackBone
        inputs = self._transform_inputs(x)
        c1, c2, c3, c4 = inputs
        b, c, h, w = c1.shape
        size = c2.size()[2:]

        # MLP + UpSampling
        c4 = self.mlp_4(c4).permute(0, 2, 1).reshape(b, -1, c4.shape[2],
                                                     c4.shape[3])
        c4 = resize(
            c4,
            size=size,
            mode='bilinear',
            align_corners=False,
        )

        c3 = self.mlp_3(c3).permute(0, 2, 1).reshape(b, -1, c3.shape[2],
                                                     c3.shape[3])
        c3 = resize(
            c3,
            size=size,
            mode='bilinear',
            align_corners=False,
        )

        c2 = self.mlp_2(c2).permute(0, 2, 1).reshape(b, -1, c2.shape[2],
                                                     c2.shape[3])

        c1 = self.mlp_1(c1).permute(0, 2, 1).reshape(b, -1, c1.shape[2],
                                                     c1.shape[3])

        c = torch.cat([c2, c3, c4], dim=1)
        c = self.cat_1(c)

        b, _, h, w = c.shape
        query = F.unfold(
            c,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        query = rearrange(
            query,
            'b (c ph pw) (nh nw) -> (b nh nw) c ph pw',
            ph=self.patch_size,
            pw=self.patch_size,
            nh=h // self.patch_size,
            nw=w // self.patch_size,
        )

        cont8, cont4, cont2 = self._get_context(c, patch_size=self.patch_size)
        output = list(
            map(
                lambda o: rearrange(
                    o,
                    '(b nh nw) c ph pw -> b c (nh ph) (nw pw)',
                    ph=self.patch_size,
                    pw=self.patch_size,
                    nh=h // self.patch_size,
                    nw=w // self.patch_size,
                ),
                [
                    self.lawin_8(query, cont8),
                    self.lawin_4(query, cont4),
                    self.lawin_2(query, cont2),
                ],
            ))
        output.extend([
            self.lawin_none(c),
            resize(
                self.image_pooling(c),
                size=(h, w),
                mode='bilinear',
                align_corners=False,
            )
        ])

        output = self.cat_2(torch.cat(output, dim=1))
        output = resize(
            output,
            size=c1.size()[2:],
            mode='bilinear',
            align_corners=False,
        )

        output = self.cat_3(torch.cat([output, c1], dim=1))

        output = self.cls_seg(output)

        return output


if __name__ == '__main__':
    head = LawinHead(
        in_channels=[64, 128, 256, 512],
        in_index=[0, 1, 2, 3],
        channels=512,
        embed_dim=256,
        num_classes=19,
    )
    h, w = 1024, 1024
    c1 = torch.zeros(1, 64, h // 4, w // 4)
    c2 = torch.zeros(1, 128, h // 8, w // 8)
    c3 = torch.zeros(1, 256, h // 16, w // 16)
    c4 = torch.zeros(1, 512, h // 32, w // 32)
    input = [c1, c2, c3, c4]
    head(input)
