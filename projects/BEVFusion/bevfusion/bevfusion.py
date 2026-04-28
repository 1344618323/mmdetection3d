from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from mmengine.utils import is_list_of
from torch import Tensor
from torch.nn import functional as F

from mmdet3d.models import Base3DDetector
from mmdet3d.registry import MODELS
from mmdet3d.structures import Det3DDataSample
from mmdet3d.utils import OptConfigType, OptMultiConfig, OptSampleList
from .ops import Voxelization


@MODELS.register_module()
class BEVFusion(Base3DDetector):

    def __init__(
        self,
        data_preprocessor: OptConfigType = None,
        pts_voxel_encoder: Optional[dict] = None,
        pts_middle_encoder: Optional[dict] = None,
        fusion_layer: Optional[dict] = None,
        img_backbone: Optional[dict] = None,
        pts_backbone: Optional[dict] = None,
        view_transform: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        pts_neck: Optional[dict] = None,
        bbox_head: Optional[dict] = None,
        init_cfg: OptMultiConfig = None,
        seg_head: Optional[dict] = None,
        **kwargs,
    ) -> None:
        voxelize_cfg = data_preprocessor.pop('voxelize_cfg')
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.voxelize_reduce = voxelize_cfg.pop('voxelize_reduce')
        self.pts_voxel_layer = Voxelization(**voxelize_cfg)

        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)

        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None
        self.img_neck = MODELS.build(
            img_neck) if img_neck is not None else None
        self.view_transform = MODELS.build(
            view_transform) if view_transform is not None else None
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)

        self.fusion_layer = MODELS.build(
            fusion_layer) if fusion_layer is not None else None

        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)

        self.bbox_head = MODELS.build(bbox_head)

        self.init_weights()

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def parse_losses(
        self, losses: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Parses the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: There are two elements. The first is the
            loss tensor passed to optim_wrapper which may be a weighted sum
            of all losses, and the second is log_vars which will be sent to
            the logger.
        """
        log_vars = []
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars.append([loss_name, loss_value.mean()])
            elif is_list_of(loss_value, torch.Tensor):
                log_vars.append(
                    [loss_name,
                     sum(_loss.mean() for _loss in loss_value)])
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(value for key, value in log_vars if 'loss' in key)
        log_vars.insert(0, ['loss', loss])
        log_vars = OrderedDict(log_vars)  # type: ignore

        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars  # type: ignore

    def init_weights(self) -> None:
        if self.img_backbone is not None:
            self.img_backbone.init_weights()

    @property
    def with_bbox_head(self):
        """bool: Whether the detector has a box head."""
        return hasattr(self, 'bbox_head') and self.bbox_head is not None

    @property
    def with_seg_head(self):
        """bool: Whether the detector has a segmentation head.
        """
        return hasattr(self, 'seg_head') and self.seg_head is not None

    def extract_img_feat(
        self,
        x,
        points,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W).contiguous()

        x = self.img_backbone(x)
        x = self.img_neck(x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        with torch.autocast(device_type='cuda', dtype=torch.float32):
            x = self.view_transform(
                x,
                points,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        return x

    def extract_pts_feat(self, batch_inputs_dict) -> torch.Tensor:
        points = batch_inputs_dict['points']
        with torch.autocast('cuda', enabled=False):
            points = [point.float() for point in points]
            feats, coords, sizes = self.voxelize(points)
            batch_size = coords[-1, 0] + 1
        x = self.pts_middle_encoder(feats, coords, batch_size)
        return x

    @torch.no_grad()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.pts_voxel_layer(res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode='constant', value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(
                    dim=1, keepdim=False) / sizes.type_as(feats).view(-1, 1)
                feats = feats.contiguous()

        return feats, coords, sizes

    def predict(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
                batch_data_samples: List[Det3DDataSample],
                **kwargs) -> List[Det3DDataSample]:
        """Forward of testing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                'points' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input sample. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
                (num_instances, )
            - labels_3d (Tensor): Labels of bboxes, has a shape
                (num_instances, ).
            - bbox_3d (:obj:`BaseInstance3DBoxes`): Prediction of bboxes,
                contains a tensor with shape (num_instances, 7).
        """
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        if self.with_bbox_head:
            outputs = self.bbox_head.predict(feats, batch_input_metas)

        res = self.add_pred_to_datasample(batch_data_samples, outputs)

        return res

    def extract_feat(
        self,
        batch_inputs_dict,
        batch_input_metas,
        **kwargs,
    ):
        imgs = batch_inputs_dict.get('imgs', None)
        points = batch_inputs_dict.get('points', None)
        features = []
        if imgs is not None:
            imgs = imgs.contiguous()
            lidar2image, camera_intrinsics, camera2lidar = [], [], []
            img_aug_matrix, lidar_aug_matrix = [], []
            for i, meta in enumerate(batch_input_metas):
                lidar2image.append(meta['lidar2img'])
                camera_intrinsics.append(meta['cam2img'])
                camera2lidar.append(meta['cam2lidar'])
                img_aug_matrix.append(meta.get('img_aug_matrix', np.eye(4)))
                lidar_aug_matrix.append(
                    meta.get('lidar_aug_matrix', np.eye(4)))

            lidar2image = imgs.new_tensor(np.asarray(lidar2image))
            camera_intrinsics = imgs.new_tensor(np.array(camera_intrinsics))
            camera2lidar = imgs.new_tensor(np.asarray(camera2lidar))
            img_aug_matrix = imgs.new_tensor(np.asarray(img_aug_matrix))
            lidar_aug_matrix = imgs.new_tensor(np.asarray(lidar_aug_matrix))
            img_feature = self.extract_img_feat(imgs, deepcopy(points),
                                                lidar2image, camera_intrinsics,
                                                camera2lidar, img_aug_matrix,
                                                lidar_aug_matrix,
                                                batch_input_metas)
            features.append(img_feature)
        pts_feature = self.extract_pts_feat(batch_inputs_dict)
        features.append(pts_feature)

        if self.fusion_layer is not None:
            x = self.fusion_layer(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return x

    def loss(self, batch_inputs_dict: Dict[str, Optional[Tensor]],
             batch_data_samples: List[Det3DDataSample],
             **kwargs) -> List[Det3DDataSample]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        feats = self.extract_feat(batch_inputs_dict, batch_input_metas)

        losses = dict()
        if self.with_bbox_head:
            bbox_loss = self.bbox_head.loss(feats, batch_data_samples)

        losses.update(bbox_loss)

        return losses

"""
---------------------------------------------------------------------
train_pipeline 对比lidar-only和lidar-cam的差异, 我们看到不同模态的数据增强方式有些不同
for lidar-only                  for lidar-cam
---------------------------------------------------------------------
                                BEVLoadMultiViewImageFromFiles
                                    这个类继承自LoadMultiViewImageFromFiles, 区别不大,
                                    父类记录了 cam2img, lidar2cam
                                    子类额外记录了 cam2lidar, lidar2img
LoadPointsFromFile              LoadPointsFromFile
LoadPointsFromMultiSweeps       LoadPointsFromMultiSweeps
LoadAnnotations3D               LoadAnnotations3D
ObjectSample
    补充采样gt和点云, 只能用于lidar-only                    
                                ImageAug3D
                                    对图像做resize, crop, flip, rotate, 并记录img_aug_matrix
GlobalRotScaleTrans             BEVFusionGlobalRotScaleTrans
                                    继承自GlobalRotScaleTrans(修改点云和GT), 多记录了lidar_aug_matrix
BEVFusionRandomFlip3D           BEVFusionRandomFlip3D
    我感觉lidar-only中直接用RandomFlip3D也行, 
    毕竟不涉及相机和lidar的转换
                                    类似RandomFlip3D, 但新增lidar_aug_matrix=R@lidar_aug_matrix
                                    R是 前后 或 左右 翻转带来的矩阵
PointsRangeFilter               PointsRangeFilter
ObjectRangeFilter               ObjectRangeFilter
ObjectNameFilter                ObjectNameFilter
                                GridMask
PointShuffle                    PointShuffle
Pack3DDetInputs                 Pack3DDetInputs

---------------------------------------------------------------------

def loss(...):
    1. feats =self.extract_feat(...):
        if lidarncam:
            img_feature = self.extract_img_feat(imgs, ...):
                x=self.img_backbone(x) mmdet.models.backbones.swin.SwinTransformer (TODO)
                    输入[B*N, C, H, W], 如 [6, 3, 256, 704]
                    输出
                        [B*N, 192, H/8, W/8], 如 [6, 192, 32, 88]
                        [B*N, 384, H/16, W/16], 如 [6, 384, 16, 44]
                        [B*N, 768, H/32, W/32], 如 [6, 768, 8, 22]
                self.img_neck(x) projects.BEVFusion.bevfusion.bevfusion_necks.GeneralizedLSSFPN, 是FPN-LSS的实现
                    输出
                        [B*N, 256, H/8, W/8], 如 [6, 256, 32, 88]
                        [B*N, 256, H/16, W/16], 如 [6, 256, 16, 44]
                x = self.view_transform(x, points, lidar2image, camera_intrinsics, camera2lidar, img_aug_matrix, lidar_aug_matrix, img_metas) 
                    projects.BEVFusion.bevfusion.bevfusion.view_transform.DepthLSSTransform
                    就是LSS中的lift+splat实现，但对bevpooling进行了优化，且参照bevdepth有引入lidar points
                    最后返回 [B, 80, 180, 180]
                return x
        pts_feature = self.extract_pts_feat(...)
            feats, coords, sizes = self.voxelize(points):
                对每个样本的点云体素化 ret = self.pts_voxel_layer(res) projects.BEVFusion.bevfusion.ops.voxel.voxelize.Voxelization
                并整合成:  
                feats [M, C] 该批次共M个体素, 每个体素使用一个C维度的特征
                coords [M, 4] 该批次共M个体素, 每个体素使用一个4维度的坐标: (batch_idx, z_idx, y_idx, x_idx)
                sizes [M] 该批次共M个体素, 每个体素内有多少个点
            x = self.pts_middle_encoder(feats, coords, batch_size) projects.BEVFusion.bevfusion.sparse_encoder.BEVFusionSparseEncoder
                其实就是SECOND的pts_middle_encoder
                输入的空间尺寸为 [1440, 1440, 41] (表示3D网格zxy方向的数量) 由[-54.0,-54.0,-5.0,54.0,54.0,3.0]/[0.075,0.075,0.2]求得
                输出的空间尺寸为 [180, 180, 2] 8倍下采样(在pts_middle_encoder中的conv_out层会对z方向再多一次2倍下采样)
                最终返回 [B, D*C, H, W] 的tensor, 举个例子: [1, 2*128, 180, 180]
            return x
        if lidarncam:
            x=fusion_layer(img_feature, pts_feature) 
                projects.BEVFusion.bevfusion.fusion_layer.FusionLayer
                过程非常简单 conv2d(cat(img_feature, pts_feature)) 输出 [B, 256, 180, 180]
        x = self.pts_backbone(x)  mmdet3d.models.backbones.second.SECOND
            其实SECOND的pts_backbone, 简单的多段2D卷积网络, 输出一个tuple:
            [B, 128, 180, 180]
            [B, 256, 90, 90]
        x = self.pts_neck(x) mmdet3d.models.necks.second_fpn.SECONDFPN
            其实就是SECOND的pts_neck, 返回一个长度为1的tuple, 元素尺寸为 [1, 512, 180, 180]
        return x
    
    2. bbox_loss = self.bbox_head.loss(feats, ...) 
        projects.BEVFusion.bevfusion.transfusion_head.TransFusionHead

    3. return bbox_loss


def parse_losses(...):
    重写了 BaseModel.parse_losses, 并没有改变算法, 只是修改了log_vars中loss的计算方式:
        在BaseModel.parse_losses, 只是统计当前rank的loss, 最后若没有啥特别配置, 在log中只输出了 rank0 的loss(不过是一段时将内的平滑值)
        在这里, 会算一个所有rank平均的loss, 最后在log中输出


def predict(...):
    1. feats =self.extract_feat(...):
    2. self.bbox_head.predict(feats, ...)
"""
