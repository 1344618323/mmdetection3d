import copy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Linear
from mmdet.models.dense_heads import DETRHead
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.utils import multi_apply
from mmdet.utils import InstanceList, OptInstanceList, reduce_mean
from mmengine.model import bias_init_with_prob
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import MODELS, TASK_UTILS
from .util import normalize_bbox


@MODELS.register_module()
class DETR3DHead(DETRHead):
    """Head of DETR3D.

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bbox_coder (obj:`ConfigDict`): Configs to build the bbox coder
        num_cls_fcs (int) : the number of layers in cls and reg branch
        code_weights (List[double]) : loss weights of
            (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y)
        code_size (int) : size of code_weights

    ------------------------------
    父类 DETRHead 源码: /opt/conda/lib/python3.8/site-packages/mmdet/models/dense_heads/detr_head.py
        插曲:
            self.positional_encoding 是 mmdet.models.layers.positional_encoding.SinePositionalEncoding
                源码: /opt/conda/lib/python3.8/site-packages/mmdet/models/layers/positional_encoding.py
                但在 DETR3DHead 中，并没有使用

    DETR3DHead 会复用部分 DETRHead.__init__ 构造的东西，但没全用:
    self.bbox_coder 是 projects.DETR3D.detr3d.nms_free_coder.NMSFreeCoder
    self.transformer 是 Detr3DTransformer
    self.assigner 是 projects.DETR3D.detr3d.hungarian_assigner_3d.HungarianAssigner3D
    self.loss_cls 是 mmdet.models.losses.focal_loss.FocalLoss 源码 /opt/conda/lib/python3.8/site-packages/mmdet/models/losses/focal_loss.py
    self.loss_bbox 是 mmdet.models.losses.smooth_l1_loss.L1Loss 源码 /opt/conda/lib/python3.8/site-packages/mmdet/models/losses/smooth_l1_loss.py
    self.loss_iou(没用，但记录下) 是 mmdet.models.losses.iou_loss.GIoULoss 是 源码 /opt/conda/lib/python3.8/site-packages/mmdet/models/losses/iou_loss.py

    DETR3DHead新加的:
    self.sampler 是 mmdet3d.models.task_modules.samplers.pseudosample.PseudoSampler
    -------

    如果配置 with_box_refine=True 且 as_two_stage = False, 则
        self.cls_branches 和 self.reg_branches 会配置独立的 6 层（因为 self.transformer 有 6 层）, 每层都是 fc_cls 和 reg_branch 的副本
        fc_cls 的结构是 Linear(256, 256), LayerNorm(256), ReLU, ..., Linear(256, num_classes)
        reg_branch 的结构是 Linear(256, 256), ReLu, ..., Linear(256, box_code_size)

    forward流程:
        hs, init_reference, inter_references = self.transformer(...)
            hs尺寸为 [6, num_query, B, embed_dims] 6是decoder transformer layer的层数, hs应该是 hidden states的缩写
            init_reference尺寸为 [B, num_query, 3]
            inter_references尺寸为 [6, B, num_query, 3]
    
        最后返回 outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        outputs_classes 尺寸为 [6, B, num_query, num_classes]
        outputs_coords 尺寸为 [6, B, num_query, code_size]
            其中 [0,1,4]是真实物理坐标系下的xyz坐标：由reg计算的偏移+reference_points得到
    
    loss_by_feat流程:
        multi_apply(self.loss_by_feat_single(...) 每个decoder layer的输出会进入
            self.get_targets(...)
                multi_apply(self._get_target_single, ...) 每个样本的输出会进入
                    最终通过HungarianAssigner3D，匹配pred与gt
                    返回 (labels, label_weights, bbox_targets, bbox_weights, pos_inds, neg_inds)
                        labels [num_query] 每个query的标签，0-based, 如果值为num_classes, 则表示背景
                        label_weights [num_query] 每个query的标签权重，1.0
                        bbox_targets [num_query, code_size] 每个query的回归目标
                        bbox_weights [num_query, code_size] 只有正样本的各个属性为1.0，其余为0
                        pos_inds [num_pos] 正样本索引
                        neg_inds [num_neg] 负样本索引
                
                self.get_targets(...)最后返回，并在self.loss_by_feat_single使用的是
                labels: [B*num_query] 每个query的标签
                label_weights: [B*num_query]
                bbox_targets: [B*num_query, code_size]
                bbox_weights: [B*num_query, code_size]
                num_total_pos: int 正样本数量
                num_total_neg: int 负样本数量

            注意到对loss有 num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item() 这个函数
            from mmdet.utils import reduce_mean 这个函数用于跨卡同步，计算平均值，为啥这样做？
            寻常做法：各卡各自算loss，各自反传算梯度，最后跨卡同步平均梯度（等价于各卡各算loss，跨卡算平均loss，反传算梯度）
            而这里会跨卡算cls_avg_factor，这是一种更“先进、现代”的归一化策略
            * 比如有的batch的pos_inds特别多，有的batch的pos_inds特别少，有的batch的pos_inds为0，
                在不跨卡的情况下，后者算loss的分母会变得极小（甚至需要加个 eps 避震），导致这帧图产生的微弱背景噪声被无限放大，产生巨大的随机梯度。
            * 跨卡同步后，即使这张卡没目标，它也会共享全局的分母。这使得“没目标”的卡能以正确的比例贡献它的背景梯度，而不会带歪整个模型。
            * 这种做法让 “多卡训练” 在数学逻辑上无限接近于 “单卡超大 Batch 训练”

            loss_cls = sigmoid focal loss，所有正样本query/负样本（背景）query都会参与
                预测张量输入为 [B*num_query, num_classes]，通道只包含前景类，不单独包含背景通道
                标签labels取值范围为 0..num_classes，其中 num_classes 表示背景
                在 mmdet.models.losses.focal_loss.FocalLoss 中 会先构造 (num_classes+1) one-hot 再截断到前 num_classes；
                即对于每个query,有 num_classes 个loss的计算; 其中，对于背景类query, num_classes 个loss全是作为负样本计算的
            
            loss_bbox = l1 loss 仅正样本query参与
        最后返回一个dict: 包含每层decoder的 loss_cls 和 loss_bbox
    
    predict_by_feat流程:
        经过前面的foward, 对每个样本得到 [num_query, num_classes] 的 cls_scores 
        (推理时decoder依然是每层transformer layer有独立的cls/reg branch, 但只用最后一层cls/reg branch的forward结果),
        通过projects.DETR3D.detr3d.nms_free_coder.NMSFreeCoder 筛选出其中分最高的bbox(没有NMS)
    """

    def __init__(
            self,
            *args,
            with_box_refine=False,
            as_two_stage=False,
            transformer=None,
            bbox_coder=None,
            num_cls_fcs=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            code_size=10,
            **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        self.code_size = code_size
        self.code_weights = code_weights

        self.bbox_coder = TASK_UTILS.build(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.num_cls_fcs = num_cls_fcs - 1
        super(DETR3DHead, self).__init__(
            *args, transformer=transformer, **kwargs)
        # DETR sampling=False, so use PseudoSampler, format the result
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = TASK_UTILS.build(sampler_cfg)

        self.code_weights = nn.Parameter(
            torch.tensor(self.code_weights, requires_grad=False),
            requires_grad=False)

    # forward_train -> loss
    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_pred = (self.transformer.decoder.num_layers + 1) if \
            self.as_two_stage else self.transformer.decoder.num_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(fc_cls, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [fc_cls for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])

        if not self.as_two_stage:
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)

    def forward(self, mlvl_feats: List[Tensor], img_metas: List[Dict],
                **kwargs) -> Dict[str, Tensor]:
        """Forward function.

        Args:
            mlvl_feats (List[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head,
                shape [nb_dec, bs, num_query, cls_out_channels]. Note
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression
                head with normalized coordinate format
                (cx, cy, l, w, cz, h, sin(φ), cos(φ), vx, vy).
                Shape [nb_dec, bs, num_query, 10].
        """
        query_embeds = self.query_embedding.weight
        hs, init_reference, inter_references = self.transformer(
            mlvl_feats,
            query_embeds,
            reg_branches=self.reg_branches if self.with_box_refine else None,
            img_metas=img_metas,
            **kwargs)
        hs = hs.permute(0, 2, 1, 3)
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)
            outputs_class = self.cls_branches[lvl](hs[lvl])
            tmp = self.reg_branches[lvl](hs[lvl])  # shape: ([B, num_q, 10])
            # TODO: check the shape of reference
            assert reference.shape[-1] == 3
            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            tmp[..., 0:1] = \
                tmp[..., 0:1] * (self.pc_range[3] - self.pc_range[0]) \
                + self.pc_range[0]
            tmp[..., 1:2] = \
                tmp[..., 1:2] * (self.pc_range[4] - self.pc_range[1]) \
                + self.pc_range[1]
            tmp[..., 4:5] = \
                tmp[..., 4:5] * (self.pc_range[5] - self.pc_range[2]) \
                + self.pc_range[2]

            # TODO: check if using sigmoid
            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        outs = {
            'all_cls_scores': outputs_classes,
            'all_bbox_preds': outputs_coords,
            'enc_cls_scores': None,
            'enc_bbox_preds': None,
        }
        return outs

    def _get_target_single(
            self,
            cls_score: Tensor,  # [query, num_cls]
            bbox_pred: Tensor,  # [query, 10]
            gt_instances_3d: InstanceList) -> Tuple[Tensor, ...]:
        """Compute regression and classification targets for a single image."""
        # turn bottm center into gravity center
        gt_bboxes = gt_instances_3d.bboxes_3d  # [num_gt, 9]
        gt_bboxes = torch.cat(
            (gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)

        gt_labels = gt_instances_3d.labels_3d  # [num_gt, num_cls]
        # assigner and sampler: PseudoSampler
        assign_result = self.assigner.assign(
            bbox_pred, cls_score, gt_bboxes, gt_labels, gt_bboxes_ignore=None)
        sampling_result = self.sampler.sample(
            assign_result, InstanceData(priors=bbox_pred),
            InstanceData(bboxes_3d=gt_bboxes))
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        num_bboxes = bbox_pred.size(0)
        labels = gt_bboxes.new_full((num_bboxes, ),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        # theta in gt_bbox here is still a single scalar
        bbox_targets = torch.zeros_like(bbox_pred)[..., :self.code_size - 1]
        bbox_weights = torch.zeros_like(bbox_pred)
        # only matched query will learn from bbox coord
        bbox_weights[pos_inds] = 1.0

        # fix empty gt bug in multi gpu training
        if sampling_result.pos_gt_bboxes.shape[0] == 0:
            sampling_result.pos_gt_bboxes = \
                sampling_result.pos_gt_bboxes.reshape(0, self.code_size - 1)

        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
                neg_inds)

    def get_targets(
            self,
            batch_cls_scores: List[Tensor],  # bs[num_q,num_cls]
            batch_bbox_preds: List[Tensor],  # bs[num_q,10]
            batch_gt_instances_3d: InstanceList) -> tuple():
        """"Compute regression and classification targets for a batch image for
        a single decoder layer.

        Args:
            batch_cls_scores (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            batch_bbox_preds (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y) and
                shape [num_query, 10]
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance.  It usually includes ``bboxes_3d``、``labels_3d``.
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
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(self._get_target_single,
                                                     batch_cls_scores,
                                                     batch_bbox_preds,
                                                     batch_gt_instances_3d)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, num_total_pos, num_total_neg)

    def loss_by_feat_single(
        self,
        batch_cls_scores: Tensor,  # bs,num_q,num_cls
        batch_bbox_preds: Tensor,  # bs,num_q,10
        batch_gt_instances_3d: InstanceList
    ) -> Tuple[Tensor, Tensor]:
        """"Loss function for outputs from a single decoder layer of a single
        feature level.

        Args:
           batch_cls_scores (Tensor): Box score logits from a single
                decoder layer for batched images with shape [num_query,
                cls_out_channels].
            batch_bbox_preds (Tensor): Sigmoid outputs from a single
                decoder layer for batched images, with normalized coordinate
                (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y) and
                shape [num_query, 10]
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d. It usually has ``bboxes_3d``,``labels_3d``.
        Returns:
            tulple(Tensor, Tensor): cls and reg loss for outputs from
                a single decoder layer.
        """
        batch_size = batch_cls_scores.size(0)  # batch size
        cls_scores_list = [batch_cls_scores[i] for i in range(batch_size)]
        bbox_preds_list = [batch_bbox_preds[i] for i in range(batch_size)]
        cls_reg_targets = self.get_targets(cls_scores_list, bbox_preds_list,
                                           batch_gt_instances_3d)

        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)

        # classification loss
        batch_cls_scores = batch_cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                batch_cls_scores.new_tensor([cls_avg_factor]))

        cls_avg_factor = max(cls_avg_factor, 1)
        loss_cls = self.loss_cls(
            batch_cls_scores, labels, label_weights, avg_factor=cls_avg_factor)

        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        # regression L1 loss
        batch_bbox_preds = batch_bbox_preds.reshape(-1,
                                                    batch_bbox_preds.size(-1))
        normalized_bbox_targets = normalize_bbox(bbox_targets, self.pc_range)
        # neg_query is all 0, log(0) is NaN
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            batch_bbox_preds[isnotnan, :self.code_size],
            normalized_bbox_targets[isnotnan, :self.code_size],
            bbox_weights[isnotnan, :self.code_size],
            avg_factor=num_total_pos)

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    # original loss()
    def loss_by_feat(
            self,
            batch_gt_instances_3d: InstanceList,
            preds_dicts: Dict[str, Tensor],
            batch_gt_instances_3d_ignore: OptInstanceList = None) -> Dict:
        """Compute loss of the head.

        Args:
            batch_gt_instances_3d (list[:obj:`InstanceData`]): Batch of
                gt_instance_3d.  It usually includes ``bboxes_3d``、`
                `labels_3d``、``depths``、``centers_2d`` and attributes.
                gt_instance.  It usually includes ``bboxes``、``labels``.
            batch_gt_instances_3d_ignore (list[:obj:`InstanceData`], Optional):
                NOT supported.
                Defaults to None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert batch_gt_instances_3d_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for batch_gt_instances_3d_ignore setting to None.'
        all_cls_scores = preds_dicts[
            'all_cls_scores']  # num_dec,bs,num_q,num_cls
        all_bbox_preds = preds_dicts['all_bbox_preds']  # num_dec,bs,num_q,10
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']

        # calculate loss for each decoder layer
        num_dec_layers = len(all_cls_scores)
        batch_gt_instances_3d_list = [
            batch_gt_instances_3d for _ in range(num_dec_layers)
        ]
        losses_cls, losses_bbox = multi_apply(self.loss_by_feat_single,
                                              all_cls_scores, all_bbox_preds,
                                              batch_gt_instances_3d_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            enc_loss_cls, enc_losses_bbox = self.loss_by_feat_single(
                enc_cls_scores, enc_bbox_preds, batch_gt_instances_3d_list)
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

    def predict_by_feat(self,
                        preds_dicts,
                        img_metas,
                        rescale=False) -> InstanceList:
        """Transform network output for a batch into bbox predictions.

        Args:
            preds_dicts (Dict[str, Tensor]):
                -all_cls_scores (Tensor): Outputs from the classification head,
                    shape [nb_dec, bs, num_query, cls_out_channels]. Note
                    cls_out_channels should includes background.
                -all_bbox_preds (Tensor): Sigmoid outputs from the regression
                    head with normalized coordinate format
                    (cx, cy, l, w, cz, h, rot_sine, rot_cosine, v_x, v_y).
                    Shape [nb_dec, bs, num_query, 10].
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Object detection results of each image
            after the post process. Each item usually contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                  (num_instances, C), where C >= 7.
        """
        # sinθ & cosθ ---> θ
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)  # batch size
        ret_list = []
        for i in range(num_samples):
            results = InstanceData()
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, self.code_size - 1)

            results.bboxes_3d = bboxes
            results.scores_3d = preds['scores']
            results.labels_3d = preds['labels']
            ret_list.append(results)
        return ret_list
