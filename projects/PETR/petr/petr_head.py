# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.layers import NormedLinear
from mmdet.models.layers.transformer import inverse_sigmoid
from mmdet.models.utils import multi_apply
from mmengine.model.weight_init import bias_init_with_prob
from mmengine.structures import InstanceData

from mmdet3d.registry import MODELS, TASK_UTILS
from projects.PETR.petr.utils import normalize_bbox


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    """
    sine/cosine position encoding. 这一步没有任何可学习参数
    sine/cosine PE 公式：
        PE(p, i) = sin(p / \omega_i) i是偶数even
        PE(p, i) = cos(p / \omega_i) i是奇数odd
    pos 位置索引；i 维度索引
    \omega_i = 10000**(2 * (i // 2) / num_pos_feats)
    num_pos_feats 是维度大小，即i的取值范围是[0, num_pos_feats-1]
    回到代码中

    进入这个函数时 pos 每个值的取值范围是[0, 1]，会转到 [0, 2pi]
    最后返回 posemb，（900, 384） 的tensor, 即900个3dpoint，每个3dpoint对应384维特征
    """
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()),
                        dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()),
                        dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()),
                        dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb


@MODELS.register_module()
class PETRHead(AnchorFreeHead):
    """Implements the DETR transformer head. See `paper: End-to-End Object
    Detection with Transformers.

    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    -------------------------------------------------------------
    self.assigner 是 projects.PETR.petr.hungarian_assigner_3d.HungarianAssigner3D 
        和 projects.DETR3D.detr3d.hungarian_assigner_3d.HungarianAssigner3D 实现是一样的
    self.sampler 是 mmdet3d.models.task_modules.samplers.pseudosample.PseudoSampler
    
    self.forward(...)
        x = self.input_proj(mlvl_feats[0]) 是 shape为 [B, N, embed_dims, H, W] 的Tensor, 如 [1, 6, 256, 20, 50]
            self.input_proj 是 Conv2d(256, 256, kernel_size=1, stride=1) 用于将fpn通道数 映射到 transformer embed_dims 的投影层

        mask大小为 [B, N, H, W], 如(B, N, 320, 800) 值为1的元素会被忽略, 值为0的元素会参与计算
            会先通过 F.interpolate 将mask大小调整为与mlvl_feats[0]一致, 如(B, N, 20, 50)
        
        获取像素3D位置编码
            coords_position_embeding = self.position_embeding(mlvl_feats) 给mlvl_feats[0]的像素坐标生成position_embeding
                (注意, 只要图像尺寸/内外参/各种配置参数不变, 那么coords_position_embeding经position_encoder前的输入coords3d是完全固定的!!!)
            sin_embed = self.positional_encoding(masks) 是shape为 [B, N, 384, H, W] 的Tensor
                self.positional_encoding 是 projects.PETR.petr.positional_encoding.SinePositionalEncoding3D 对象,
                    其实现似乎与 mmdet.models.layers.positional_encoding.SinePositionalEncoding 有些不同, 
                    具体没仔细看, 似乎添加了3D方面的东西, 有空再说吧.
                    不过二者都没有可学习参数
            sin_embed 经过 self.adapt_pos3d(1*1卷积->relu->1*1卷积), 得到sin_embed, 是shape为 [B, N, embed_dims, H, W] 的Tensor, 如 [1, 6, 256, 20, 50]
            pos_embed = sin_embed + coords_position_embeding 得到最后的PE, shape为 [B, N, embed_dims, H, W]
        
        获取query位置编码
            self.reference_points: nn.Embedding(self.num_query 即900, 3)，其weight会通过[0,1]均一分布初始化 
            query_embeds = self.query_embedding(pos2posemb3d(reference_points))
                经过 sine/cosine position encoding 得到 [900, 384] 的tensor
                再经过 self.query_embedding(1*1卷积->relu->1*1卷积), 得到 [900, 256] 的tensor
                因此 query_embeds 是 [900, 256] 的tensor
            reference_points 复制B份，得到 [B, 900, 3] tensor

            关于query PE, 论文的消融实验 Query Generator 一节中提到了4种设计。
                碎碎念，虽然论文中叫query生成，但在代码中其实是指query PE生成
                四种实现，除了第一种，其他三种的实现方式都是我问AI的，论文并没有具体说
            1. Learned-3D, 即代码中的实现方式：
                1.1. reference_points 来自 nn.Embedding. 其中 reference_points 是 可学习的, 会先用[0,1]均一分布初始化
                1.2. query PE = MLP(sine/cosine PE(reference_points))
            2. None
                2.1 query PE 来自 nn.Embedding
            3. Fix-BEV
                3.1 reference_points 取自一个xy网格，均一分布，z取一固定值
                3.2 query PE = MLP(sine/cosine PE(reference_points))
            4. Fix-3D
                4.1 reference_points 取自一个xyz网格，均一分布
                4.2 query PE = MLP(sine/cosine PE(reference_points))
            论文的消融实验结果：Learned-3D 效果最好

        outs_dec = self.transformer(x, mask, query_embeds, pos_embed)
            self.transformer 是 projects.PETR.petr.petr_transformer.PETRTransformer 的obj, 其forward方法会返回 out_dec, memory
            输入：
                x: mlvl_feats[0]通过self.input_proj投影后 [B, N, embed_dims, H, W] tensor ，作为cross_attn的key和value.
                    而在petrhead中，mlvl_feats[1]是没有参与的
                mask: 作为 multiheadattention 的 key_padding_mask, [B, N, H, W] 的tensor, 值为1的元素会被忽略, 值为0的元素会参与计算
                query_embeds: [num_query, embed_dims] 的tensor，作为query的PE
                pos_embed: [B, N, embed_dims, H, W] 的 tensor， 作为key的PE
                另外，最初的query是 shape为[num_query, embed_dims] 的 0 tensor
                
            输出：outs_dec 是 [num_layers, B, num_query, embed_dims] 的tensor, 即6层decoder的输出

        outs_dec 分别经过 self.cls_branches 和 self.reg_branches 得到 cls_scores 和 bbox_preds，
        最后返回
        outs = {
            'all_cls_scores': 长度为num_pred的list, 每个元素是 [B, num_query, num_classes] 的tensor
            'all_bbox_preds': 长度为num_pred的list, 每个元素是 [B, num_query, code_size] 的tensor
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        self.cls_branches 和 self.reg_branches 配置和 DETR3DHead 是一样的

    self.loss_by_feat(...): 这一部分和DETR3D是一样的
        multi_apply(self.loss_by_feat_single, ...) 分别处理一个decoder layer的输出:
            self.get_targets(...):
                multi_apply(self._get_target_single, ...): 分别匹配每一个样本的gt和det
            每个decoder layer都会使用self.loss_cls(FocalLoss) 和 self.loss_bbox(L1Loss) 计算loss
        最后返回 loss_dict

    self.get_bboxes(...): 这一部分和DETR3D是一样的

    -------------------------------------------------------
    attention map 可视化
    论文的visualization一节提到了attention map可视化，我很好奇是怎么实现的，拷打了下AI，结论如下：
    在最后一层decoder layer的cross-attn中：
    query [num_query, B, embed_dims]
    key [N*H*W, B, embed_dims]
    有 attention score = softmax(query * key^T / sqrt(embed_dims)), 是一个 [num_query, B, N*H*W] 的tensor
    即每个样本的每个query对应一个 N*H*W 的tensor，对于N个相机，我们可以将其上采样到原图尺寸，可视化出来。
    
    但注意，通常我们用的是多头attention，可视化的attn map 一般是 多头 attn map 的平均。
    就像torch API
    attn = nn.MultiheadAttention(
        embed_dim=256,   # 输入特征的总维度 (例如 256)
        num_heads=8,     # 注意力头的数量 (必须能被 embed_dim 整除)
        ...
    )
    attn_output, attn_output_weights = attn(
                query=query,
                key=key,
                value=value,
                average_attn_weights=True
    )
    输出的attn_output_weights 是一个 [num_query, B, N*H*W] 的tensor
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                     assigner=dict(
                         type='HungarianAssigner',
                         cls_cost=dict(type='ClassificationCost', weight=1.),
                         reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                         iou_cost=dict(
                             type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 with_position=True,
                 with_multiview=False,
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start=1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 init_cfg=None,
                 normedlinear=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [
                1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2
            ]
        self.code_weights = self.code_weights[:self.code_size]
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is PETRHead):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']
            assert loss_cls['loss_weight'] == assigner['cls_cost']['weight'], \
                'The classification weight for loss and matcher should be' \
                'exactly the same.'
            assert loss_bbox['loss_weight'] == assigner['reg_cost'][
                'weight'], 'The regression L1 weight for loss and matcher ' \
                'should be exactly the same.'
            # assert loss_iou['loss_weight'] == assigner['iou_cost'][
            #   'weight'], \
            # 'The regression iou weight for loss and matcher should be' \
            # 'exactly the same.'
            self.assigner = TASK_UTILS.build(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = TASK_UTILS.build(sampler_cfg)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = 256
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = 3 * self.depth_num
        self.position_range = position_range
        self.LID = LID
        self.depth_start = depth_start
        self.position_level = 0
        self.with_position = with_position
        self.with_multiview = with_multiview
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        super(PETRHead, self).__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            bbox_coder=bbox_coder,
            init_cfg=init_cfg)

        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_iou = MODELS.build(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        # self.activate = build_activation_layer(self.act_cfg)
        # if self.with_multiview or not self.with_position:
        #     self.positional_encoding = build_positional_encoding(
        #         positional_encoding)
        self.positional_encoding = TASK_UTILS.build(positional_encoding)
        self.transformer = MODELS.build(transformer)
        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False),
            requires_grad=False)
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the transformer head."""
        if self.with_position:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        else:
            self.input_proj = Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(
                NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        if self.with_multiview:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims * 3 // 2,
                    self.embed_dims * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(
                    self.embed_dims * 4,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0),
            )
        else:
            self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(
                    self.embed_dims,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(
                    self.embed_dims,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0),
            )

        if self.with_position:
            self.position_encoder = nn.Sequential(
                nn.Conv2d(
                    self.position_dim,
                    self.embed_dims * 4,
                    kernel_size=1,
                    stride=1,
                    padding=0),
                nn.ReLU(),
                nn.Conv2d(
                    self.embed_dims * 4,
                    self.embed_dims,
                    kernel_size=1,
                    stride=1,
                    padding=0),
            )

        self.reference_points = nn.Embedding(self.num_query, 3)
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3 // 2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        self.transformer.init_weights()
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def position_embeding(self, img_feats, img_metas, masks=None):
        """
        作用：给mlvl_feats[0]的每个像素生成position_embeding

        对于depth的采样, CaDNN 和 PETR 在取bin时, 有几种不同配置
        if self.LID: 
            使用LID, Linear Increasing Discretization 线性递增离散化, 深度方向上的 bin 宽度随深度大致线性增大 (近处更密、远处更疏)
            第1个bin, 长度为 d, 第2个bin长度为 2d, ..., 第n个bin长度为 n*d
            假如有n个bin, 总长为 (1+n)*n/2 *d = dmax-dmin
            d = (dmax-dmin) / ((1+n)*n/2)
            截至到第i个bin, 总长为 d+2d+...+id + dmin = d*i*(i+1)/2 + dmin = i(i+1) * (dmax-dmin)/(n(n+1)) + dmin
        else:
            则采用 UD (Uniform Discretization 均匀离散化), 即深度bin是均匀的, 等于 (dmax-dmin)/n
            第1个bin, 长度为 d, 第2个bin长度为 d, ..., 第n-1个bin长度为 d
            截至到第i个bin, 总长为 i * d + dmin = i/n * (dmax-dmin) + dmin

        论文的消融实验中对PETR with LID、PETR with UD做了比较，发现效果差不多

        回到代码中

        最后获取的coords, 是shape为 [B, N, W, H, D, 4, 1] 的Tensor, 如 [1, 6, 50, 20, 64, 4, 1]
            W, H, D 分别对应图像的宽度(0, 16, 32, ..., 800), 高度(0, 16, 32, ..., 320), 深度(1, 1.0289, 1.0868, ..., 59,3477)
            最后一维4, 是齐次坐标, 表示某一个像素坐标对应的 (Du, Dv, D, 1)
        而coords3d = img2lidars @ coords, 是shape为 [B, N, W, H, D, 3] 的Tensor, 如 [1, 6, 50, 20, 64, 3]
            最后一维3, 即对应每一个像素坐标的lidar3D坐标, 如 (x, y, z), 会归一化到 [0, 1] 范围
        
        给coord3d 在像素层面打码:
            对于最后一维的xyz, 要在[0,1]范围内. 对每一个像素有 D 个bin, 一共 (D, 3)个坐标.
            如果 (D, 3) 个值中在[0, 1]范围内的数量 > 0.5D, 则认为该像素有效, 保留
            最后 coords_mask 和 mask 与操作, 得到 [B, N, H, W] 的 tensor
            值得一提的是, coords_mask 压根没用,不知道是不是bug? 
        
        coords3d 会转成 [B*N, D*3, H, W] 的tensor, 如 [1*6, 64*3, 20, 50]
            并使用inverse_sigmoid, 将(0,1)映射到(-inf, inf)

        值得一提的是, 只要 img_feats 的尺寸、img_metas 中的内外参与配置参数保持不变，
            coords3d 的输出就是完全固定（确定性）的!!!
        
        coord3d 经过 self.position_encoder(1*1卷积->relu->1*1卷积), 得到coords_position_embeding, 
            是shape为 [B*N, embed_dims, H, W] 的Tensor, 如 [6, 256, 20, 50]的tensor,
            即每个mlvl_feat的像素坐标, 都对应一个 embed_dims 维的向量

        论文的消融实验中特意提到了 self.position_encoder 的设计, 有三种设计
        1. 无
        2. 1*1卷积->relu->1*1卷积
        3. 3*3卷积->...
        方案2相比方案1，mAP/NDS 均有提升
        方案3相比方案1都是副作用，导致mAP/NDS接近0。论文认为3*3卷积会破坏 3D坐标与2D像素的关联

        另外论文还讨论了 key 和 keyPE 的融合方式：
        1. add
        2. concat
        3. dot product
        实验结果： add/concat 效果差不多，dot product 效果最差
        """
        eps = 1e-5
        pad_h, pad_w = img_metas[0]['pad_shape']
        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(
            H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(
            W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index = torch.arange(
                start=0,
                end=self.depth_num,
                step=1,
                device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (
                self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index = torch.arange(
                start=0,
                end=self.depth_num,
                step=1,
                device=img_feats[0].device).float()
            bin_size = (self.position_range[3] -
                        self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d
                                             ])).permute(1, 2, 3,
                                                         0)  # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3],
            torch.ones_like(coords[..., 2:3]) * eps)

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars)  # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4,
                                     4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (
            self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (
            self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (
            self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0)
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3,
                                    2).contiguous().view(B * N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(coords3d)

        return coords_position_embeding.view(B, N, self.embed_dims, H,
                                             W), coords_mask

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is PETRHead:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)

    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format \
                (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """

        x = mlvl_feats[0]
        batch_size, num_cams = x.size(0), x.size(1)
        input_img_h, input_img_w = img_metas[0]['pad_shape']
        masks = x.new_ones((batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
        x = self.input_proj(x.flatten(0, 1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(masks, size=x.shape[-2:]).to(torch.bool)

        """
        self.with_position, self.with_multiview 要结合论文的消融实验来看，用于决定给mlvl_feats[0]的每个像素生成

        self.with_position: 
            true: 给mlvl_feats[0]的每个像素生成3DPE
            false: 不生成3DPE
        self.with_multiview: 
            true: 给mlvl_feats[0]的每个像素生成2DPE 考虑N维度
            false: 各个相机分别给mlvl_feats[0]的每个像素生成2DPE，再concat到一起
        最后PE=3DPE+2DPE

        实验结果： with_position=True, with_multiview=True 时效果最好，但是主要提升还是来自3DPE
        """
        
        if self.with_position:
            coords_position_embeding, _ = self.position_embeding(
                mlvl_feats, img_metas, masks)
            pos_embed = coords_position_embeding
            if self.with_multiview:
                sin_embed = self.positional_encoding(masks)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(
                    x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.positional_encoding(masks[:, i:i+1, :, :])
                    pos_embeds.append(xy_embed)
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(
                    x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.with_multiview:
                pos_embed = self.positional_encoding(masks)
                pos_embed = self.adapt_pos3d(pos_embed.flatten(0, 1)).view(
                    x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.positional_encoding(masks[:, i:i+1, :, :])
                    pos_embeds.append(pos_embed)
                pos_embed = torch.cat(pos_embeds, 1)

        reference_points = self.reference_points.weight
        query_embeds = self.query_embedding(pos2posemb3d(reference_points))
        reference_points = reference_points.unsqueeze(0).repeat(
            batch_size, 1, 1)  # .sigmoid()

        outs_dec, _ = self.transformer(x, masks, query_embeds, pos_embed,
                                       self.reg_branches)
        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl]).to(
                torch.float32)
            tmp = self.reg_branches[lvl](outs_dec[lvl]).to(torch.float32)

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        all_bbox_preds[..., 0:1] = (
            all_bbox_preds[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) +
            self.pc_range[0])
        all_bbox_preds[..., 1:2] = (
            all_bbox_preds[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) +
            self.pc_range[1])
        all_bbox_preds[..., 4:5] = (
            all_bbox_preds[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) +
            self.pc_range[2])

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """

        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        assign_result = self.assigner.assign(bbox_pred, cls_score, gt_bboxes,
                                             gt_labels, gt_bboxes_ignore)
        pred_instance_3d = InstanceData(priors=bbox_pred)
        gt_instances_3d = InstanceData(bboxes_3d=gt_bboxes)
        sampling_result = self.sampler.sample(assign_result, pred_instance_3d,
                                              gt_instances_3d)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(bbox_pred)[..., :code_size]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0
        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]
        gt_labels_list = gt_labels_list[0]
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list,
         neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                      bbox_preds_list, gt_labels_list,
                                      gt_bboxes_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_by_feat_single(self,
                            cls_scores,
                            bbox_preds,
                            gt_bboxes_list,
                            gt_labels_list,
                            gt_bboxes_ignore_list=None):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
            cls_scores (Tensor): Box score logits from a single decoder layer
                for all images. Shape [bs, num_query, cls_out_channels].
            bbox_preds (Tensor): Sigmoid outputs from a single decoder layer
                for all images, with normalized coordinate (cx, cy, w, h) and
                shape [bs, num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in
                [tl_x, tl_y, br_x,loss_by_feat_single br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs
                from a single decoder layer.
        ------------------------------------------------------------
        gt_bboxes_list 中各维度含义： cx cy cz l w h yaw vx vy
        bbox_preds_list 中各维度含义： cx cy logl logw cz logh sin(yaw) cos(yaw) vx vy
        但不要担心:
        1. 在self.assigner.assign(...) 中算loss时，会将 gt_bboxes_list 转成
            cx cy logl logw cz logh sin(yaw) cos(yaw) vx vy
        2. self.loss_bbox前，也会就将 gt_bboxes_list 转成
            cx cy logl logw cz logh sin(yaw) cos(yaw) vx vy
        值得一提的是，这个 sin/cos 内其实还有一个 CCW（逆时针） <-> CW（顺时针） 的转换，具体见 utils.py 中的 normalize_bbox 和 denormalize_bbox
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        bbox_preds_list = [bbox_preds[i] for i in range(num_imgs)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           gt_bboxes_list, gt_labels_list,
                                           gt_bboxes_ignore_list)
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)

        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        # if self.sync_cls_avg_factor:
        #     cls_avg_factor = reduce_mean(
        #         cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        # num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        num_total_pos = torch.clamp(num_total_pos, min=1).item()

        # regression L1 loss
        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :10],
            normalized_bbox_targets[isnotnan, :10],
            bbox_weights[isnotnan, :10],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def loss_by_feat(self,
                     gt_bboxes_list,
                     gt_labels_list,
                     preds_dicts,
                     gt_bboxes_ignore=None):
        """"Loss function.
        Args:
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device

        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]),
                      dim=1).to(device) for gt_bboxes in gt_bboxes_list
        ]

        all_gt_bboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [[gt_labels_list] for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]

        losses_cls, losses_bbox = multi_apply(self.loss_by_feat_single,
                                              all_cls_scores, all_bbox_preds,
                                              all_gt_bboxes_list,
                                              all_gt_labels_list,
                                              all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(all_gt_labels_list))
            ]
            enc_loss_cls, enc_losses_bbox = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox

        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i in zip(losses_cls[:-1], losses_bbox[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
