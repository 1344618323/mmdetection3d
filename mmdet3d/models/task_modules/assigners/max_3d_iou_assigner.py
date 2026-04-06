# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Union

from mmdet.models.task_modules import AssignResult, MaxIoUAssigner
from mmengine.structures import InstanceData

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class Max3DIoUAssigner(MaxIoUAssigner):
    # TODO: This is a temporary box assigner.
    """Assign a corresponding gt bbox or background to each bbox.

    Each proposals will be assigned with `-1`, or a semi-positive integer
    indicating the ground truth index.

    - -1: negative sample, no assigned gt
    - semi-positive integer: positive sample, index (0-based) of assigned gt

    Args:
        pos_iou_thr (float): IoU threshold for positive bboxes.
        neg_iou_thr (float or tuple): IoU threshold for negative bboxes.
        min_pos_iou (float): Minimum iou for a bbox to be considered as a
            positive bbox. Positive samples can have smaller IoU than
            pos_iou_thr due to the 4th step (assign max IoU sample to each gt).
            `min_pos_iou` is set to avoid assigning bboxes that have extremely
            small iou with GT as positive samples.
        gt_max_assign_all (bool): Whether to assign all bboxes with the same
            highest overlap with some gt to that gt.
        ignore_iof_thr (float): IoF threshold for ignoring bboxes (if
            `gt_bboxes_ignore` is specified). Negative values mean not
            ignoring any bboxes.
        ignore_wrt_candidates (bool): Whether to compute the iof between
            `bboxes` and `gt_bboxes_ignore`, or the contrary.
        match_low_quality (bool): Whether to allow low quality matches. This is
            usually allowed for RPN and single stage detectors, but not allowed
            in the second stage. Details are demonstrated in Step 4.
        gpu_assign_thr (int): The upper bound of the number of GT for GPU
            assign. When the number of gt is above this threshold, will assign
            on CPU device. Negative values mean not assign on CPU.
        iou_calculator (dict): Config of overlaps Calculator.

    函数注释有错误, 以我的解读为准
    -1 忽略
    0 背景(负样本)
    1-N 正样本(值为GT的1-based索引,1对应GT[0],2对应GT[1],...)

    self.iou_calculator 的类型可以是 self.iou_calculator BboxOverlapsNearest3D

    总结: 使用gt bev aabb1 和 pred bev aabb2 计算2D IoU.
    然后通过iou阈值做gt和pred的匹配. 可能出现多个pred匹配到同一个gt 或者 一个gt没有被任何pred匹配到 的情况

    --------------------------------------------------------
    self.assign_wrt_overlaps 源码在 /opt/conda/lib/python3.8/site-packages/mmdet/models/task_modules/assigners/max_iou_assigner.py 中

    overlaps (k, n) k个gt, n个pred的IoU
    max_overlaps, argmax_overlaps 每个pred对应的最大iou和其对应的gt索引
    gt_max_overlaps, gt_argmax_overlaps 每个gt对应的最大iou和其对应的pred索引

    step1: assigned_gt_inds[i] = -1, 即每个pred都是忽略的

    step2: 
    对于 0<=max_overlaps[i]<neg_iou_thr, 则 assigned_gt_inds[i] = 0, 即该pred是负样本
    或者是 neg_iou_thr[0] <= max_overlaps[i] < neg_iou_thr[1], 则 assigned_gt_inds[i] = 0, 
        也就说对于iou特别低的pred, 我们会将其只其忽略,而不是作为负样本

    step3:
    对于max_overlaps[i]>=pos_iou_thr, 则 assigned_gt_inds[i] = argmax_overlaps[i] + 1, 即该pred是正样本

    IoU:-inf    neg_thr[0]    neg_thr[1]   pos_thr         1
    举例值           0          0.3          0.6
        |-----------|-----------|-----------|-------------|
          忽略(-1)     负样本(0)    忽略(-1)     正样本(>0)

    可以发现 argmax_overlaps[i] 的赋值是不排它的, 一个gt可能被多个pred匹配到

    step4:
    若使能了self.match_low_quality
    则对于每个gt, 若gt_max_overlaps[i]>=self.min_pos_iou(如0.3),则将与其有最大iou的pred赋值为i+1, 即该gt是正样本.
    这么做的目的是为了尽可能多地让gt被匹配到

    返回: AssignResult(
        num_gts=num_gts, 即gt的个数
        gt_inds=assigned_gt_inds, 即每个pred对应的gt索引, -1表示忽略, 0表示背景, >0表示正样本
        max_overlaps=max_overlaps, 即每个pred对应的最大iou
        labels=assigned_labels 即每个pred对应的gt标签, -1表示没有(没有区分忽略和背景), >=0 表示类型(0-based)
    )
    --------------------------------------------------------
    """

    def __init__(
        self,
        pos_iou_thr: float,
        neg_iou_thr: Union[float, tuple],
        min_pos_iou: float = .0,
        gt_max_assign_all: bool = True,
        ignore_iof_thr: float = -1,
        ignore_wrt_candidates: bool = True,
        match_low_quality: bool = True,
        gpu_assign_thr: float = -1,
        iou_calculator: dict = dict(type='BboxOverlaps2D')
    ) -> None:
        self.pos_iou_thr = pos_iou_thr
        self.neg_iou_thr = neg_iou_thr
        self.min_pos_iou = min_pos_iou
        self.gt_max_assign_all = gt_max_assign_all
        self.ignore_iof_thr = ignore_iof_thr
        self.ignore_wrt_candidates = ignore_wrt_candidates
        self.gpu_assign_thr = gpu_assign_thr
        self.match_low_quality = match_low_quality
        self.iou_calculator = TASK_UTILS.build(iou_calculator)

    def assign(self,
               pred_instances: InstanceData,
               gt_instances: InstanceData,
               gt_instances_ignore: Optional[InstanceData] = None,
               **kwargs) -> AssignResult:
        """Assign gt to bboxes.

        This method assign a gt bbox to every bbox (proposal/anchor), each bbox
        will be assigned with -1, or a semi-positive number. -1 means negative
        sample, semi-positive number is the index (0-based) of assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every bbox to the background
        2. assign proposals whose iou with all gts < neg_iou_thr to 0
        3. for each bbox, if the iou with its nearest gt >= pos_iou_thr,
           assign it to that bbox
        4. for each gt bbox, assign its nearest proposals (may be more than
           one) to itself

        Args:
            pred_instances (:obj:`InstanceData`): Instances of model
                predictions. It includes ``priors``, and the priors can
                be anchors or points, or the bboxes predicted by the
                previous stage, has shape (n, 4). The bboxes predicted by
                the current model or stage will be named ``bboxes``,
                ``labels``, and ``scores``, the same as the ``InstanceData``
                in other places.
            gt_instances (:obj:`InstanceData`): Ground truth of instance
                annotations. It usually includes ``bboxes``, with shape (k, 4),
                and ``labels``, with shape (k, ).
            gt_instances_ignore (:obj:`InstanceData`, optional): Instances
                to be ignored during training. It includes ``bboxes``
                attribute data that is ignored during training and testing.
                Defaults to None.

        Returns:
            :obj:`AssignResult`: The assign result.

        Example:
            >>> from mmengine.structures import InstanceData
            >>> self = MaxIoUAssigner(0.5, 0.5)
            >>> pred_instances = InstanceData()
            >>> pred_instances.priors = torch.Tensor([[0, 0, 10, 10],
            ...                                      [10, 10, 20, 20]])
            >>> gt_instances = InstanceData()
            >>> gt_instances.bboxes = torch.Tensor([[0, 0, 10, 9]])
            >>> gt_instances.labels = torch.Tensor([0])
            >>> assign_result = self.assign(pred_instances, gt_instances)
            >>> expected_gt_inds = torch.LongTensor([1, 0])
            >>> assert torch.all(assign_result.gt_inds == expected_gt_inds)
        """
        gt_bboxes = gt_instances.bboxes_3d
        if 'priors' in pred_instances:
            priors = pred_instances.priors
        else:
            priors = pred_instances.bboxes_3d.tensor
        gt_labels = gt_instances.labels_3d
        if gt_instances_ignore is not None:
            gt_bboxes_ignore = gt_instances_ignore.bboxes_3d
        else:
            gt_bboxes_ignore = None

        assign_on_cpu = True if (self.gpu_assign_thr > 0) and (
            gt_bboxes.shape[0] > self.gpu_assign_thr) else False
        # compute overlap and assign gt on CPU when number of GT is large
        if assign_on_cpu:
            device = priors.device
            priors = priors.cpu()
            gt_bboxes = gt_bboxes.cpu()
            gt_labels = gt_labels.cpu()
            if gt_bboxes_ignore is not None:
                gt_bboxes_ignore = gt_bboxes_ignore.cpu()

        overlaps = self.iou_calculator(gt_bboxes, priors)

        if (self.ignore_iof_thr > 0 and gt_bboxes_ignore is not None
                and gt_bboxes_ignore.numel() > 0 and priors.numel() > 0):
            if self.ignore_wrt_candidates:
                ignore_overlaps = self.iou_calculator(
                    priors, gt_bboxes_ignore, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=1)
            else:
                ignore_overlaps = self.iou_calculator(
                    gt_bboxes_ignore, priors, mode='iof')
                ignore_max_overlaps, _ = ignore_overlaps.max(dim=0)
            overlaps[:, ignore_max_overlaps > self.ignore_iof_thr] = -1

        assign_result = self.assign_wrt_overlaps(overlaps, gt_labels)
        if assign_on_cpu:
            assign_result.gt_inds = assign_result.gt_inds.to(device)
            assign_result.max_overlaps = assign_result.max_overlaps.to(device)
            if assign_result.labels is not None:
                assign_result.labels = assign_result.labels.to(device)
        return assign_result
