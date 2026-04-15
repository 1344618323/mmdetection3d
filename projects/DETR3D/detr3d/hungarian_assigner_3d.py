from typing import List

import torch
from mmdet.models.task_modules.assigners import AssignResult  # check
from mmdet.models.task_modules.assigners import BaseAssigner
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet3d.registry import TASK_UTILS
from .util import normalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None


@TASK_UTILS.register_module()
class HungarianAssigner3D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of some components.
    For DETR3D the costs are weighted sum of classification cost, regression L1
    cost and regression iou cost. The targets don't include the no_object, so
    generally there are more predictions than targets. After the one-to-one
    matching, the un-matched are treated as backgrounds. Thus each query
    prediction will be assigned with `0` or a positive integer indicating the
    ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        cls_cost (obj:`ConfigDict`) : Match cost configs.
        reg_cost.
        iou_cost.
        pc_range: perception range of the detector

    --------------------------------
    self.cls_cost 是 mmdet.models.task_modules.assigners.match_cost.FocalLossCost
        源码 /opt/conda/lib/python3.8/site-packages/mmdet/models/task_modules/assigners/match_cost.py
        作用：输入(cls_pred, gt_labels)
            cls_pred 尺寸为 [num_query, num_classes]; gt_labels 尺寸为 [num_gt]取值为 0到num_classes-1
            返回 (num_query, num_gt) 的 cost matrix, 如何计算的呢？
            计算每个query 作为第i（取值范围为0到num_classes-1）类的正样本的FocalLossCost： pos_cost，即一个[num_query, num_classes]的向量
            再计算每个query 作为第i类的负样本的FocalLossCost： neg_cost，即一个[num_query, num_classes]的矩阵
                
                focalloss公式= -(\alpha*y+(1-\alpha)*(1-p_t))^{\gamma}log(p_t)
                y=1时，即作为正样本时 pos_cost=-(\alpha)*(1-p)^{\gamma}log(p)
                y=0时，即作为负样本时 neg_cost=-(1-\alpha)*(p)^{\gamma}log(1-p)

            最后 cls_cost = pos_cost - neg_cost
                表示将一个query从 i类 转为 非i类 的cost，cost越小越容易匹配

            我们只关注num_classes这些类中，属于gt_labels的那些类，即有 cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
                一个 [num_query, num_gt] 的矩阵
            所以 cost[j, i] 含义就是 索引为j的query 与 索引为i的gt 类别 匹配的cost

    self.reg_cost 是 projects.DETR3D.detr3d.match_cost.BBox3DL1Cost
        torch.cdist(bbox_pred, gt_bboxes, p=1) 计算 bbox_pred 和 gt_bboxes 两两之间的 L1 距离
            cost(i,j)= \sum_{k=0}^{8} |bbox_pred[i][k] - gt_bboxes[j][k]|, 包括xyz log(lwh) sin(rot) cos(rot)
        
        k=0,1,2,3,4,5,6,7 分别对应 x,y,z,l,w,h,sin(rot),cos(rot)

    ---
    self.iou_cost （没有用） 是 mmdet.models.task_modules.assigners.match_cost.IoUCost
        源码 /opt/conda/lib/python3.8/site-packages/mmdet/models/task_modules/assigners/match_cost.py
    ---

    最后用于计算匈牙利匹配的cost矩阵：
        cost = cls_cost + reg_cost, 前者权重默认2，后者权重默认0.25
    使用 from scipy.optimize import linear_sum_assignment 求解，cost越小越容易匹配
    
    最后返回 AssignResult，包含：
        num_gts: 目标框数量
        assigned_gt_inds: 每个预测框分配的gt索引，0表示背景(作为样本)，正数表示gt索引(1-based)
            注意这个+1只是约定，源码 /opt/conda/lib/python3.8/site-packages/mmdet/models/task_modules/samplers/sampling_result.py 中
                可以看到 self.pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1，即又减回去了
        labels: 每个预测框分配的gt类别，-1表示未分配，>=0表示gt类别
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range: List = None):
        self.cls_cost = TASK_UTILS.build(cls_cost)
        self.reg_cost = TASK_UTILS.build(reg_cost)
        self.iou_cost = TASK_UTILS.build(iou_cost)
        self.pc_range = pc_range

    def assign(self,
               bbox_pred: Tensor,
               cls_pred: Tensor,
               gt_bboxes: Tensor,
               gt_labels: Tensor,
               gt_bboxes_ignore=None,
               eps=1e-7) -> AssignResult:
        """Computes one-to-one matching based on the weighted costs.
        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.
        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx,cy,l,w,cz,h,sin(φ),cos(φ),v_x,v_y) which are all in
                range [0, 1] and shape [num_query, 10].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (cx,cy,cz,l,w,h,φ,v_x,v_y). Shape [num_gt, 9].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): unused parameter
        Returns:
            :obj:`AssignResult`: The assigned result.
        """
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)  # 9, 900

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        # 2. compute the weighted costs
        # classification and bboxcost.
        # # dev1.x interface alignment
        pred_instances = InstanceData(scores=cls_pred)
        gt_instances = InstanceData(labels=gt_labels)
        cls_cost = self.cls_cost(pred_instances, gt_instances)
        # regression L1 cost
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])

        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)
