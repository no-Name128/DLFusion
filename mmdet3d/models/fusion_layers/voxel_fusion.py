import torch
from mmcv.cnn import ConvModule, xavier_init, Conv2d, build_conv_layer
from mmcv.runner import auto_fp16
from torch import nn as nn
from torch.nn import functional as F

from ..registry import FUSION_LAYERS
import termcolor
from . import apply_3d_transformation
from ..utils import ImgVoxelProj
from .. import builder


@FUSION_LAYERS.register_module()
class VoxelFusion(nn.Module):
    def __init__(self,
                 unified_conv=None,
                 view_cfg=None,
                 depth_head=None,
                 img_voxel_fuser=None,
                 collapse_pts_in=1280,
                 collapse_pts_out=256,
                 **kwargs):
        super(VoxelFusion, self).__init__()
        self.unified_conv = unified_conv
        self.depth_head = depth_head
        self.view_cfg = view_cfg
        self.collapse_pts = kwargs['collapse_pts']

        """ img 2d feats to 3d voxel feats """
        if view_cfg is not None:
            self.img_voxel_trans = ImgVoxelProj(**view_cfg)

        if self.unified_conv is not None:
            self.conv_layer = []
            for k in range(self.unified_conv['num_conv']):
                conv = nn.Sequential(
                    nn.Conv3d(kwargs['embed_dims_for_fuser'],
                              kwargs['embed_dims_for_fuser'],
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              bias=True),
                    nn.BatchNorm3d(kwargs['embed_dims_for_fuser']),
                    nn.ReLU(inplace=True))
                self.add_module("{}_head_{}".format('conv_trans', k + 1), conv)
                self.conv_layer.append(conv)

        # config for the depth pred for the images multi-views
        self.input_proj = Conv2d(self.depth_head.in_channels, self.depth_head.out_channels, kernel_size=1)
        if "SimpleDepth" in depth_head.type:
            self.depth_dim = depth_head.model.depth_dim
            self.depth_net = Conv2d(self.depth_head.out_channels, self.depth_dim, kernel_size=1)   # depth_dim : 64
        else:
            raise NotImplementedError
        self.depth_head = depth_head

        if not self.collapse_pts:
            self.channel_reduce = nn.Sequential(
                build_conv_layer(
                    dict(type='Conv2d', bias=False),
                    collapse_pts_in,
                    collapse_pts_out,
                    1,
                ),
                nn.BatchNorm2d(collapse_pts_out, eps=0.001, momentum=0.01, affine=True, track_running_stats=True),
                nn.ReLU(),
            )

        if img_voxel_fuser:
            self.img_voxel_fuser = builder.build_fusion_layer(img_voxel_fuser)

    @auto_fp16(apply_to=('img'))
    def pred_depth(self, img, img_metas, img_feats=None):
        if img_feats is None:
            return None

        B = img.size(0)
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            # 将nuscenes或者是其他数据中 一个frame中的多张/单张图片进行转换，即B*N 的大小，方便卷积处理。
            img = img.view(B * N, C, H, W)
        if self.depth_head.type == "SimpleDepth":
            depth = []
            # img_feats : b n c h w
            for _feat in img_feats:
                _depth = self.depth_net(_feat.view(-1, *_feat.shape[-3:]))  # n：6 c h w
                _depth = _depth.softmax(dim=1)
                depth.append(_depth)
        else:
            raise NotImplementedError
        return depth

    def forward(self, img, img_input, pts_input, img_metas):
        """ Extracts features of depth from the imgs  """
        img_depth = self.pred_depth(img=img, img_metas=img_metas, img_feats=img_input)

        B = pts_input[0].size()[0]

        if self.depth_head is not None:
            img_feats_multi_views = []
            for img_feat in img_input:
                img_feat = self.input_proj(img_feat)
                bv, c, h, w = img_feat.size()
                img_feats_multi_views.append(img_feat.view(B, int(bv / B), c, h, w))

        """ 2D feats to 3D voxel feats for img multi-views """
        img_voxel_feats = self.img_voxel_trans(img_feats_multi_views, img_metas=img_metas, img_depth=img_depth)
        if img_voxel_feats.dim() > 5:
            if img_voxel_feats.size()[1] == 1:
                img_voxel_feats = img_voxel_feats.squeeze(1)

        if img_voxel_feats.dim() == pts_input[0].dim():
            if self.unified_conv is not None:
                pts_shape = pts_input[0].shape
                if pts_input[0].dim() == 5 and img_voxel_feats.dim() == 5:
                    # cat_feats = pts_input[0] + img_voxel_feats
                    cat_feats = self.img_voxel_fuser(img_voxel_feats, pts_input[0])
                elif pts_input[0].dim() > 5 and img_voxel_feats.dim() > 5:
                    # cat_feats = pts_input[0].flatten(1, 2) + img_voxel_feats.flatten(1, 2)
                    cat_feats = self.img_voxel_fuser(img_voxel_feats.flatten(1, 2), pts_input[0].flatten(1, 2))
                else:
                    raise NotImplementedError

                for layer in self.conv_layer:
                    cat_feats = layer(cat_feats)
                    
                if self.img_voxel_fuser:
                    cat_feats = cat_feats.reshape(pts_shape[0], self.img_voxel_fuser.out_channels, *pts_shape[2:])
                else:
                    cat_feats = cat_feats.reshape(*pts_shape)

                return cat_feats
        else:
            raise NotImplementedError
