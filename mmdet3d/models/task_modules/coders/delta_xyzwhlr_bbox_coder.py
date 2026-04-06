# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmdet.models.task_modules import BaseBBoxCoder
from torch import Tensor

from mmdet3d.registry import TASK_UTILS


@TASK_UTILS.register_module()
class DeltaXYZWLHRBBoxCoder(BaseBBoxCoder):
    """Bbox Coder for 3D boxes.

    Args:
        code_size (int): The dimension of boxes to be encoded.
    """

    def __init__(self, code_size: int = 7) -> None:
        super(DeltaXYZWLHRBBoxCoder, self).__init__()
        self.code_size = code_size

    @staticmethod
    def encode(src_boxes: Tensor, dst_boxes: Tensor) -> Tensor:
        """Get box regression transformation deltas (dx, dy, dz, dx_size,
        dy_size, dz_size, dr, dv*) that can be used to transform the
        `src_boxes` into the `target_boxes`.

        Args:
            src_boxes (torch.Tensor): source boxes, e.g., object proposals.
            dst_boxes (torch.Tensor): target of the transformation, e.g.,
                ground-truth boxes.

        Returns:
            torch.Tensor: Box transformation deltas.

        src_boxes模型预测的boxes, dst_boxes 是gt的boxes, 要求二者等长
        值得一提的是
        1. za是anchor的底面z值
        2. 解码/编码时，对xyzlwh的偏移量都进行了归一化，都是基于anchor先验size做的
        
        xt = xg-xa/diagonal = (xg-xa)/sqrt(wa^2 + la^2)
        
        为什么这样?这是为了让回归目标归一化，消除尺度差异的影响。举个例子
        大卡车：anchor 尺寸 10m × 4m，GT 偏移了 1m
        小行人：anchor 尺寸 0.5m × 0.5m，GT 偏移了 0.1m
        如果直接算差值 xt = xg - xa：
        卡车：xt = 1.0
        行人：xt = 0.1
        网络需要为不同大小的物体输出差异很大的回归值，这对训练不友好。

        lt = log(lg/la)

        旋转差距 是直接的 rt = rg - ra, 没有考虑角度周期性, 但不用担心，
            后续算loss前 会用 Anchor3DHead.add_sin_difference 重置旋转差，来考虑角度周期性
        """
        box_ndim = src_boxes.shape[-1]
        cas, cgs, cts = [], [], []
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(
                src_boxes, 1, dim=-1)
            xg, yg, zg, wg, lg, hg, rg, *cgs = torch.split(
                dst_boxes, 1, dim=-1)
            cts = [g - a for g, a in zip(cgs, cas)]
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(src_boxes, 1, dim=-1)
            xg, yg, zg, wg, lg, hg, rg = torch.split(dst_boxes, 1, dim=-1)
        za = za + ha / 2
        zg = zg + hg / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha
        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)
        rt = rg - ra
        return torch.cat([xt, yt, zt, wt, lt, ht, rt, *cts], dim=-1)

    @staticmethod
    def decode(anchors: Tensor, deltas: Tensor) -> Tensor:
        """Apply transformation `deltas` (dx, dy, dz, dx_size, dy_size,
        dz_size, dr, dv*) to `boxes`.

        Args:
            anchors (torch.Tensor): Parameters of anchors with shape (N, 7).
            deltas (torch.Tensor): Encoded boxes with shape
                (N, 7+n) [x, y, z, x_size, y_size, z_size, r, velo*].

        Returns:
            torch.Tensor: Decoded boxes.

        -----------------------
        解码：
        xa,ya,za,wa,la,ha,ra,*cas都是预设的anchor的参数
        xt,yt,zt,wt,lt,ht,rt是模型预测的偏移量
        """
        cas, cts = [], []
        box_ndim = anchors.shape[-1]
        if box_ndim > 7:
            xa, ya, za, wa, la, ha, ra, *cas = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt, *cts = torch.split(deltas, 1, dim=-1)
        else:
            xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
            xt, yt, zt, wt, lt, ht, rt = torch.split(deltas, 1, dim=-1)

        za = za + ha / 2
        diagonal = torch.sqrt(la**2 + wa**2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha
        rg = rt + ra
        zg = zg - hg / 2
        cgs = [t + a for t, a in zip(cts, cas)]
        return torch.cat([xg, yg, zg, wg, lg, hg, rg, *cgs], dim=-1)
