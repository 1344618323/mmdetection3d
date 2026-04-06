# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmdet.models.utils import select_single_mlvl
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, constant_init
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.structures import limit_period, xywhr2xyxyr
from mmdet3d.structures.det3d_data_sample import SampleList
from mmdet3d.utils.typing_utils import InstanceList, OptMultiConfig


class Base3DDenseHead(BaseModule, metaclass=ABCMeta):
    """Base class for 3D DenseHeads.

    1. The ``init_weights`` method is used to initialize densehead's
    model parameters. After detector initialization, ``init_weights``
    is triggered when ``detector.init_weights()`` is called externally.

    2. The ``loss`` method is used to calculate the loss of densehead,
    which includes two steps: (1) the densehead model performs forward
    propagation to obtain the feature maps (2) The ``loss_by_feat`` method
    is called based on the feature maps to calculate the loss.

    .. code:: text

    loss(): forward() -> loss_by_feat()

    3. The ``predict`` method is used to predict detection results,
    which includes two steps: (1) the densehead model performs forward
    propagation to obtain the feature maps (2) The ``predict_by_feat`` method
    is called based on the feature maps to predict detection results including
    post-processing.

    .. code:: text

    predict(): forward() -> predict_by_feat()

    4. The ``loss_and_predict`` method is used to return loss and detection
    results at the same time. It will call densehead's ``forward``,
    ``loss_by_feat`` and ``predict_by_feat`` methods in order.  If one-stage is
    used as RPN, the densehead needs to return both losses and predictions.
    This predictions is used as the proposal of roihead.

    .. code:: text

    loss_and_predict(): forward() -> loss_by_feat() -> predict_by_feat()
    """

    def __init__(self, init_cfg: OptMultiConfig = None) -> None:
        super().__init__(init_cfg=init_cfg)

    def init_weights(self) -> None:
        """Initialize the weights."""
        super().init_weights()
        # avoid init_cfg overwrite the initialization of `conv_offset`
        for m in self.modules():
            # DeformConv2dPack, ModulatedDeformConv2dPack
            if hasattr(m, 'conv_offset'):
                constant_init(m.conv_offset, 0)

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             **kwargs) -> dict:
        """Perform forward propagation and loss calculation of the detection
        head on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.

        以pointpillars为例
        输入x为
        [
            [B, C, ny, nx] 如 [1, 256, 200, 200]
            [B, C, ny/2, nx/2]
            [B, C, ny/4, nx/4]
        ]
        设定num_classes = 10, box_code_size = 9, num_rot = 2, num_size = 4 (num_anchors = num_rot * num_size = 8)
        outs 是个tuple, (cls_scores, bbox_preds, dir_cls_preds)
        cls_scores 为 [[B, num_anchors * num_classes, ny, nx], [B, num_anchors * num_classes, ny/2, nx/2], [B, num_anchors * num_classes, ny/4, nx/4]]
        bbox_preds 为 [[B, num_anchors * box_code_size, ny, nx], [B, num_anchors * box_code_size, ny/2, nx/2], [B, num_anchors * box_code_size, ny/4, nx/4]]
        dir_cls_preds 为 [[B, num_anchors * 2, ny, nx], [B, num_anchors * 2, ny/2, nx/2], [B, num_anchors * 2, ny/4, nx/4]]
        
        具体数值案例
        -----
        torch.Size([1, 80, 200, 200])
        torch.Size([1, 80, 100, 100])
        torch.Size([1, 80, 50, 50])
        -----
        torch.Size([1, 72, 200, 200])
        torch.Size([1, 72, 100, 100])
        torch.Size([1, 72, 50, 50])
        -----
        torch.Size([1, 16, 200, 200])
        torch.Size([1, 16, 100, 100])
        torch.Size([1, 16, 50, 50])
        -----

        关注下data_sample, 即一帧的标注信息
        metainfo 中有 图像尺寸, 点云范围, 增强参数, 坐标转换
        gt_instances_3d 中是 list [3D框, 类别]

        最后的loss_inputs是个长度为6的tuple, 分别是
        (cls_scores, bbox_preds, dir_cls_preds, batch_gt_instances_3d, batch_input_metas, batch_gt_instances_ignore)
        """
        outs = self(x)

        batch_gt_instances_3d = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        loss_inputs = outs + (batch_gt_instances_3d, batch_input_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    @abstractmethod
    def loss_by_feat(self, **kwargs) -> dict:
        """Calculate the loss based on the features extracted by the detection
        head."""
        pass

    def loss_and_predict(self,
                         x: Tuple[Tensor],
                         batch_data_samples: SampleList,
                         proposal_cfg: Optional[ConfigDict] = None,
                         **kwargs) -> Tuple[dict, InstanceList]:
        """Perform forward propagation of the head, then calculate loss and
        predictions from the features and data samples.

        Args:
            x (tuple[Tensor]): Features from FPN.
            batch_data_samples (list[:obj:`Det3DDataSample`]): Each item
                contains the meta information of each image and
                corresponding annotations.
            proposal_cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.

        Returns:
            tuple: the return value is a tuple contains:

                - losses: (dict[str, Tensor]): A dictionary of loss components.
                - predictions (list[:obj:`InstanceData`]): Detection
                  results of each image after the post process.
        """
        batch_gt_instances = []
        batch_gt_instances_ignore = []
        batch_input_metas = []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances.append(data_sample.gt_instances_3d)
            batch_gt_instances_ignore.append(
                data_sample.get('ignored_instances', None))

        outs = self(x)

        loss_inputs = outs + (batch_gt_instances, batch_input_metas,
                              batch_gt_instances_ignore)
        losses = self.loss_by_feat(*loss_inputs)

        predictions = self.predict_by_feat(
            *outs, batch_input_metas=batch_input_metas, cfg=proposal_cfg)
        return losses, predictions

    def predict(self,
                x: Tuple[Tensor],
                batch_data_samples: SampleList,
                rescale: bool = False) -> InstanceList:
        """Perform forward propagation of the 3D detection head and predict
        detection results on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_pts_panoptic_seg` and
                `gt_pts_sem_seg`.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        outs = self(x)
        predictions = self.predict_by_feat(
            *outs, batch_input_metas=batch_input_metas, rescale=rescale)
        return predictions

    def predict_by_feat(self,
                        cls_scores: List[Tensor],
                        bbox_preds: List[Tensor],
                        dir_cls_preds: List[Tensor],
                        batch_input_metas: Optional[List[dict]] = None,
                        cfg: Optional[ConfigDict] = None,
                        rescale: bool = False,
                        **kwargs) -> InstanceList:
        """Transform a batch of output features extracted from the head into
        bbox results.

        Args:
            cls_scores (list[Tensor]): Classification scores for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for all
                scale levels, each is a 4D-tensor, has shape
                (batch_size, num_priors * 4, H, W).
            score_factors (list[Tensor], optional): Score factor for
                all scale level, each is a 4D-tensor, has shape
                (batch_size, num_priors * 1, H, W). Defaults to None.
            batch_input_metas (list[dict], Optional): Batch inputs meta info.
                Defaults to None.
            cfg (ConfigDict, optional): Test / postprocessing
                configuration, if None, test_cfg would be used.
                Defaults to None.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            list[:obj:`InstanceData`]: Detection results of each sample
            after the post process.
            Each item usually contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (BaseInstance3DBoxes): Prediction of bboxes,
              contains a tensor with shape (num_instances, C), where
              C >= 7.
        
        -----------------------
        输入的是一个批次的featuremap

        每个样本的预测结果会分别进入 _predict_by_feat_single 处理
        cls_score_list 是list, 长度为num_levels, 每个元素是 (num_anchors * num_classes, H, W), 其中 num_anchors = num_rot * num_size
        bbox_pred_list 是list, 长度为num_levels, 每个元素是 (num_anchors * box_code_size, H, W)
        dir_cls_pred_list 是list, 长度为num_levels, 每个元素是 (num_anchors * 2, H, W)
        
        最后得到的是一个list, 长度为batch_size, 每个元素是 InstanceData（表示一个样本的预测结果）
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_priors = self.prior_generator.grid_anchors(
            featmap_sizes, device=cls_scores[0].device)
        mlvl_priors = [
            prior.reshape(-1, self.box_code_size) for prior in mlvl_priors
        ]

        result_list = []

        for input_id in range(len(batch_input_metas)):

            input_meta = batch_input_metas[input_id]
            cls_score_list = select_single_mlvl(cls_scores, input_id)
            bbox_pred_list = select_single_mlvl(bbox_preds, input_id)
            dir_cls_pred_list = select_single_mlvl(dir_cls_preds, input_id)

            results = self._predict_by_feat_single(
                cls_score_list=cls_score_list,
                bbox_pred_list=bbox_pred_list,
                dir_cls_pred_list=dir_cls_pred_list,
                mlvl_priors=mlvl_priors,
                input_meta=input_meta,
                cfg=cfg,
                rescale=rescale,
                **kwargs)
            result_list.append(results)
        return result_list

    def _predict_by_feat_single(self,
                                cls_score_list: List[Tensor],
                                bbox_pred_list: List[Tensor],
                                dir_cls_pred_list: List[Tensor],
                                mlvl_priors: List[Tensor],
                                input_meta: dict,
                                cfg: ConfigDict,
                                rescale: bool = False,
                                **kwargs) -> InstanceData:
        """Transform a single points sample's features extracted from the head
        into bbox results.

        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single point cloud sample, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single point cloud sample, each item
                has shape (num_priors * C, H, W).
            dir_cls_pred_list (list[Tensor]): Predictions of direction class
                from all scale levels of a single point cloud sample, each
                item has shape (num_priors * 2, H, W).
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid. In all
                anchor-based methods, it has shape (num_priors, 4). In
                all anchor-free methods, it has shape (num_priors, 2)
                when `with_stride=True`, otherwise it still has shape
                (num_priors, 4).
            input_meta (dict): Contain point clouds and image meta info.
            cfg (:obj:`ConfigDict`): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Defaults to False.

        Returns:
            :obj:`InstanceData`: Detection results of each image
            after the post process.
            Each item usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                  (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                  (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                  the last dimension 4 arrange as (x1, y1, x2, y2).

        -----------------------
        总的来说，这个函数就是通过score max+ nms（2dbbox带旋转） 筛选出pred的bbox

        在zip(cls_score_list, ...)中，会分别对每个level的预测结果做处理
        mlvl_bboxes 是list, 长度为num_levels, 每个元素是 (num_anchors, box_code_size)
            来源：对于每个level,会分别求每个anchor分最高的那个类，即(320000, 10)->(32000, );再算其中最高分的那 nms_pre 个anchor,即(1000,)
        
        mlvl_bboxes: 即所有层，分最高的那(num_levels*nms_pre) 个bbox, 尺寸(num_levels*nms_pre, box_code_size)
        mlvl_bboxes_for_nms: (num_levels*nms_pre, 5) 5维是(x1,y1,x2,y2,r)
        mlvl_scores: (num_levels*nms_pre, cls+1) 最后一维是背景类
        mlvl_dir_scores: (num_levels*nms_pre, ) 对于每个bbox，会取其最高的那个方向, 虽然变量名写的是score，但实际是高分的那个索引，即方向

        mlvl是multi-level的缩写

        在box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms, ...)中，会先在各个类内做nms，然后concat到一起
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_priors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, priors in zip(
                cls_score_list, bbox_pred_list, dir_cls_pred_list,
                mlvl_priors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                priors = priors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(priors, bbox_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)

        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr, cfg.max_num,
                                       cfg, mlvl_dir_scores)
        bboxes, scores, labels, dir_scores = results
        if bboxes.shape[0] > 0:
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)
            bboxes[..., 6] = (
                dir_rot + self.dir_offset +
                np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        results = InstanceData()
        results.bboxes_3d = bboxes
        results.scores_3d = scores
        results.labels_3d = labels

        return results

    # TODO: Support augmentation test
    def aug_test(self,
                 aug_batch_feats,
                 aug_batch_input_metas,
                 rescale=False,
                 with_ori_nms=False,
                 **kwargs):
        pass
