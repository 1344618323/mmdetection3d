# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import List, Tuple

import numpy as np
import torch
from mmdet.models.utils import multi_apply
from mmdet.utils.memory import cast_tensor_type
from mmengine.runner import amp
from torch import Tensor
from torch import nn as nn

from mmdet3d.models.task_modules import PseudoSampler
from mmdet3d.models.test_time_augs import merge_aug_bboxes_3d
from mmdet3d.registry import MODELS, TASK_UTILS
from mmdet3d.utils.typing_utils import (ConfigType, InstanceList,
                                        OptConfigType, OptInstanceList)
from .base_3d_dense_head import Base3DDenseHead
from .train_mixins import AnchorTrainMixin


@MODELS.register_module()
class Anchor3DHead(Base3DDenseHead, AnchorTrainMixin):
    """Anchor-based head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        init_cfg (dict or list[dict], optional): Initialization config dict.

    ------------------------------------------------------------
    以pointpillars为例
    self.prior_generator 是 mmdet3d.models.task_modules.anchor.anchor_3d_generator.AlignedAnchor3DRangeGenerator: 用于给featuremap生成所有anchor的向量
    self.bbox_assigner 是 mmdet3d.models.task_modules.assigners.max_3d_iou_assigner.Max3DIoUAssigner
    self.bbox_sampler 是 mmdet3d.models.task_modules.samplers.pseudosample.PseudoSampler 
    self.bbox_coder 是 mmdet3d.models.task_modules.coders.delta_xyzwhlr_bbox_coder.DeltaXYZWLHRBBoxCoder
    
    self.loss_cls 是 mmdet.FocalLoss 源码/opt/conda/lib/python3.8/site-packages/mmdet/models/losses/focal_loss.py
        Focal Loss 核心公式 FL(p_t)=-\alpha_t(1-p_t)^{\gamma}log(p_t)
        p_t = y*p 其中y是gt的label, p是模型预测的概率, 即 pt = p if y=1 else 1-p
        \alpha_t, \gamma 是超参数, 通常 \alpha_t=0.25, \gamma=2.0
        
        mmdet.FocalLoss 源码中 使用的是 sigmoid_focal_loss, 即 每个类别独立做二分类, C个类别分别产生C个loss.
            其中 p = 1/(1+exp(-x)). 对于N个anchor, sigmoid激活后有 (N,C)的tensor, 经过平均得到 标量loss
        与之对应的是 softmax_focal_loss, 即 所有类别一起做多分类, 所有类别产生一个loss
            其中 p = exp(x_i)/sum(exp(x_k)), k=1,2,...,C, i是label标签

        这是pointpillars 中用 PseudoSampler 不做正负样本采样的原因: Focal Loss 的 gamma 机制自动解决了正负不平衡，不需要显式地控制正负比例。
    
    self.loss_bbox 是 mmdet.SmoothL1Loss 源码/opt/conda/lib/python3.8/site-packages/mmdet/models/losses/smooth_l1_loss.py
        SmoothL1Loss 核心公式 
            L = 0.5*(x-y)^2/\beta, if |x-y| < \beta 当误差小时, 用平方项(L2), 梯度平滑趋于0, 避免在接近目标时振荡
            L = |x-y| - 0.5/\beta, otherwise 当误差大时, 用绝对值项(L1), 梯度为+-1, 避免大误差产生过大梯度导致训练不稳
            \beta 是超参数, 通常 \beta=1.0

    self.loss_dir 是 mmdet.CrossEntropyLoss 源码/opt/conda/lib/python3.8/site-packages/mmdet/models/losses/cross_entropy_loss.py
        核心公式 -log(Softmax(x)_{label_class})

    最后对于pointpillars而言: 
    loss_cls: 忽略样本权重为0, 正负样本有权重, 负样本算作背景类, 最后所有anchor的loss/正样本数量。
        为何不是除（正样本+负样本）？因为绝大多数负样本是简单负样本, loss近乎于0, 除(正样本+负样本)会把正样本的loss稀释掉
    loss_box= 仅正样本的loss/正样本数量
    loss_dir= 仅正样本的loss/正样本数量

    ------------------------------------------------------------
    调用逻辑
    Base3DDenseHead.loss(x, batch_data_samples, ...)
        Anchor3DHead.forward(x): multi_apply(self.forward_single, x) 
            即每个level的featuremap 分别过 Anchor3DHead.forward_single (分别是3个1*1的2d卷积), 得到 
                cls_score [B,num_anchors * num_classes,H,W], 其中num_anchors=num_rot * num_size, num_classes是类别数
                bbox_pred [B,num_anchors * box_code_size,H,W]
                dir_cls_pred [B,num_anchors * 2,H,W]
            最后返回 featuremap1的结果，featuremap2的结果，featuremap3的结果..., 每个结果都是 (cls_score, bbox_pred, dir_cls_pred) 的tuple
        Anchor3DHead.loss_by_feat(loss_input, batch_gt...): 
            anchor_list = self.get_anchors(...) 生成anchor. 返回 长度为B的列表, 每个元素也是个列表, 子列表长度为feature map的level数量, 孙子列表的每个元素是 [D,H,W,nums_sizes,R,>=7] 的tensor
            cls_reg_targets = self.anchor_target_3d(...) 其实就是调用 AnchorTrainMixin.anchor_target_3d(...)，匹配anchor和gt：
                1. 将属于同一样本的不同level的anchor叠在一起，因为gt与anchor匹配时，允许一个gt被多个anchor匹配到
                2. 每个样本分别调用 multi_apply(AnchorTrainMixin.anchor_target_3d_single, ...)
                    即每个样本调用 AnchorTrainMixin.anchor_target_single_assigner(...)
                        1. bbox_assigner.assign(pred, gt, ...) 
                            这里有一点要注意：无论什么样本，无论迭代多少次，pred（即anchors）的值都是一样的.
                            通过aabb 2d iou 来划分anchor中的正样本，负样本，忽略样本
                        2. bbox_sampler.sample(...) 在pointpillars中没有用
                        3. bbox_coder.encode(...) 为正样本的anchor编码到gt的回归目标：dx,dy,dz都是会尺寸归一化，dl,dw,dh都会log归一化，dr是直接的差距（但后续会通过sin考虑周期性）
                        4. get_direction_target(...) 为正样本编码方向分类目标(朝前/朝后)
                        5. 正样本的anchor 设置 0-based labels, 负样本设置为背景类；正、负样本有权重，忽略样本0权重
                3. 划分成F（featuremap level）,并返回
            multi_apply(self._loss_by_feat_single, ...)
                即每个level的featuremap 分别过 self._loss_by_feat_single, 得到 loss_cls, loss_bbox, loss_dir
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 feat_channels: int = 256,
                 use_direction_classifier: bool = True,
                 anchor_generator: ConfigType = dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[3.9, 1.6, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size: bool = False,
                 assign_per_class: bool = False,
                 diff_rad_by_sin: bool = True,
                 dir_offset: float = -np.pi / 2,
                 dir_limit_offset: int = 0,
                 bbox_coder: ConfigType = dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='mmdet.SmoothL1Loss',
                     beta=1.0 / 9.0,
                     loss_weight=2.0),
                 loss_dir: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss', loss_weight=0.2),
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptConfigType = None) -> None:
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        warnings.warn(
            'dir_offset and dir_limit_offset will be depressed and be '
            'incorporated into box coder in the future')

        # build anchor generator
        self.prior_generator = TASK_UTILS.build(anchor_generator)
        # In 3D detection, the anchor stride is connected with anchor size
        # 其大小为num_base_anchors = num_rot * num_size 预设的rot数量*预设的size数量
        self.num_anchors = self.prior_generator.num_base_anchors
        # build box coder
        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in [
            'mmdet.FocalLoss', 'mmdet.GHMC'
        ]
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_dir = MODELS.build(loss_dir)

        self._init_layers()
        self._init_assigner_sampler()

        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = TASK_UTILS.build(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = TASK_UTILS.build(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                TASK_UTILS.build(res) for res in self.train_cfg.assigner
            ]

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                          self.num_anchors * 2, 1)

    def forward_single(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Forward function on a single-scale feature map.

        Args:
            x (Tensor): Features of a single scale level.

        Returns:
            tuple:
                cls_score (Tensor): Cls scores for a single scale level
                    the channels number is num_base_priors * num_classes.
                bbox_pred (Tensor): Box energies / deltas for a single scale
                    level, the channels number is num_base_priors * C.
                dir_cls_pred (Tensor | None): Direction classification
                    prediction for a single scale level, the channels
                    number is num_base_priors * 2.
        """
        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        dir_cls_pred = None
        if self.use_direction_classifier:
            dir_cls_pred = self.conv_dir_cls(x)
        return cls_score, bbox_pred, dir_cls_pred

    def forward(self, x: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward pass.

        Args:
            x (tuple[Tensor]): Features from the upstream network,
                each is a 4D-tensor.

        Returns:
            tuple: A tuple of classification scores, bbox and direction
                classification prediction.

                - cls_scores (list[Tensor]): Classification scores for all
                    scale levels, each is a 4D-tensor, the channels number
                    is num_base_priors * num_classes.
                - bbox_preds (list[Tensor]): Box energies / deltas for all
                    scale levels, each is a 4D-tensor, the channels number
                    is num_base_priors * C.
                - dir_cls_preds (list[Tensor|None]): Direction classification
                    predictions for all scale levels, each is a 4D-tensor,
                    the channels number is num_base_priors * 2.
        
        可以看下 multi_apply 的实现, 很有意思
        x是 i0, i1, i2, 有 
        (o00, o01, o02) = self.forward_single(i0), (o10, o11, o12) = self.forward_single(i1), (o20, o21, o22) = self.forward_single(i2)
        return ([(o00, o10, o20), (o01, o11, o21), (o02, o12, o22)])
        """
        return multi_apply(self.forward_single, x)

    # TODO: Support augmentation test
    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_input_metas,
                 rescale=False,
                 **kwargs):
        aug_bboxes = []
        # only support aug_test for one sample
        for x, input_meta in zip(aug_batch_feats, aug_batch_input_metas):
            outs = self.forward(x)
            bbox_list = self.get_results(*outs, [input_meta], rescale=rescale)
            bbox_dict = dict(
                bboxes_3d=bbox_list[0].bboxes_3d,
                scores_3d=bbox_list[0].scores_3d,
                labels_3d=bbox_list[0].labels_3d)
            aug_bboxes.append(bbox_dict)
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, aug_batch_input_metas,
                                            self.test_cfg)
        return [merged_bboxes]

    def get_anchors(self,
                    featmap_sizes: List[tuple],
                    input_metas: List[dict],
                    device: str = 'cuda') -> list:
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            input_metas (list[dict]): contain pcd and img's meta info.
            device (str): device of current module.

        Returns:
            list[list[torch.Tensor]]: Anchors of each image, valid flags
                of each image.

        返回长度为B的列表, 列表每个元素也是个列表, 子列表长度为feature map的level数量, 子列表的每个元素是 [D,H,W,nums_sizes,R,>=7] 的tensor
        """
        num_imgs = len(input_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.prior_generator.grid_anchors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        return anchor_list

    def _loss_by_feat_single(self, cls_score: Tensor, bbox_pred: Tensor,
                             dir_cls_pred: Tensor, labels: Tensor,
                             label_weights: Tensor, bbox_targets: Tensor,
                             bbox_weights: Tensor, dir_targets: Tensor,
                             dir_weights: Tensor, num_total_samples: int):
        """Calculate loss of Single-level results.

        Args:
            cls_score (Tensor): Class score in single-level.
            bbox_pred (Tensor): Bbox prediction in single-level.
            dir_cls_pred (Tensor): Predictions of direction class
                in single-level.
            labels (Tensor): Labels of class.
            label_weights (Tensor): Weights of class loss.
            bbox_targets (Tensor): Targets of bbox predictions.
            bbox_weights (Tensor): Weights of bbox loss.
            dir_targets (Tensor): Targets of direction predictions.
            dir_weights (Tensor): Weights of direction loss.
            num_total_samples (int): The number of valid samples.

        Returns:
            tuple[torch.Tensor]: Losses of class, bbox
                and direction, respectively.

        输入:
        cls_score (B,num_anchors * num_classes,H,W) num_anchors=num_rot * num_size 一个网格内的anchor数量
        bbox_pred (B,num_anchors * box_code_size,H,W) box_code_size=7+dv*
        dir_cls_pred (B,num_anchors * R,H,W)
        labels (B,N) N=num_anchors*H*W
        label_weights (B,N)
        bbox_targets (B,N,>=7)
        bbox_weights (B,N,>=7)
        dir_targets (B,N)
        dir_weights (B,N)
        num_total_samples 是该batch所有featuremap下的正样本anchor数量(对于self.sampling=False的情况, 
            对于self.sampling=True的情况, 是该batch所有featuremap下的正样本anchor数量和负样本anchor数量之和,
            对于pointpillars, self.sampling=False)

        在进入self.loss_cls前
        labels (B*N)
        label_weights (B*N,)
        cls_score (B*N,num_classes)
        最后 loss_cls 是标量

        在进入self.loss_bbox前
        bbox_pred (B*N,box_code_size)
        bbox_targets (B*N,box_code_size)
        bbox_weights (B*N,box_code_size)
        最后 loss_bbox 是标量

        在进入self.loss_dir前
        dir_cls_pred (B*N,2)
        dir_targets (B*N)
        dir_weights (B*N)
        最后 loss_dir 是标量
        """
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        assert labels.max().item() <= self.num_classes
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(
                        as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        # dir loss
        if self.use_direction_classifier:
            dir_cls_pred = dir_cls_pred.permute(0, 2, 3, 1).reshape(-1, 2)
            dir_targets = dir_targets.reshape(-1)
            dir_weights = dir_weights.reshape(-1)
            pos_dir_cls_pred = dir_cls_pred[pos_inds]
            pos_dir_targets = dir_targets[pos_inds]
            pos_dir_weights = dir_weights[pos_inds]

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            if self.diff_rad_by_sin:
                pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                    pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=num_total_samples)

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                loss_dir = self.loss_dir(
                    pos_dir_cls_pred,
                    pos_dir_targets,
                    pos_dir_weights,
                    avg_factor=num_total_samples)
        else:
            loss_bbox = pos_bbox_pred.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_pred.sum()

        return loss_cls, loss_bbox, loss_dir

    @staticmethod
    def add_sin_difference(boxes1: Tensor, boxes2: Tensor) -> tuple:
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
                dimensions are changed.
        
        predbbox[6]' = sin(predbbox[6])*cos(gtbbox[6])
        gtbbox[6]' = cos(predbbox[6])*sin(gtbbox[6])
        注意 predbbox[6] 是模型预测值, 而 gtbbox[6] = rg-ra, rg是gt的yaw角度, ra是anchor的yaw角度
        如此一来, 在使用smoothl1时, 就有
        sin(predbbox[6])*cos(gtbbox[6])-cos(predbbox[6])*sin(gtbbox[6])=sin(predbbox[6]-gtbbox[6])
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    def loss_by_feat(
            self,
            cls_scores: List[Tensor],
            bbox_preds: List[Tensor],
            dir_cls_preds: List[Tensor],
            batch_gt_instances_3d: InstanceList,
            batch_input_metas: List[dict],
            batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instances. It usually includes ``bboxes_3d``
                and ``labels_3d`` attributes.
            batch_input_metas (list[dict]): Contain pcd and img's meta info.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of gt_instances_ignore. It includes ``bboxes`` attribute
                data that is ignored during training and testing.
                Defaults to None.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification
                    losses.

        最后返回字典:
        losses_cls 对于val: 长度为F的列表, 每个元素是标量, F是feature map的level数量
        losses_bbox 对于val: 长度为F的列表, 每个元素是标量
        losses_dir 对于val: 长度为F的列表, 每个元素是标量
        """
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.prior_generator.num_levels
        device = cls_scores[0].device
        anchor_list = self.get_anchors(
            featmap_sizes, batch_input_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            batch_gt_instances_3d,
            batch_input_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            num_classes=self.num_classes,
            label_channels=label_channels,
            sampling=self.sampling)

        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # num_total_samples = None
        with amp.autocast(enabled=False):
            losses_cls, losses_bbox, losses_dir = multi_apply(
                self._loss_by_feat_single,
                cast_tensor_type(cls_scores, dst_type=torch.float32),
                cast_tensor_type(bbox_preds, dst_type=torch.float32),
                cast_tensor_type(dir_cls_preds, dst_type=torch.float32),
                labels_list,
                label_weights_list,
                bbox_targets_list,
                bbox_weights_list,
                dir_targets_list,
                dir_weights_list,
                num_total_samples=num_total_samples)
        return dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)
