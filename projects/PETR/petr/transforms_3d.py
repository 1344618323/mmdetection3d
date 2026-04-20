# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.transforms import BaseTransform
from PIL import Image

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import LiDARInstance3DBoxes


@TRANSFORMS.register_module()
class ResizeCropFlipImage(BaseTransform):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.
    """

    def __init__(self, data_aug_conf=None, training=True):
        self.data_aug_conf = data_aug_conf
        self.training = training

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.

        所有view的图像使用同样的线性变换: 一个受约束的仿射变换,更准确的是 含翻转的相似变换
        样本中的img和相机内参 会在这个函数中被变换
        之所以不同view使用同样的线性变换, 是为了保持不同view之间的几何关系一致
        """

        imgs = results['img']
        N = len(imgs)
        new_imgs = []
        resize, resize_dims, crop, flip, rotate = self._sample_augmentation()
        results['lidar2cam'] = np.array(results['lidar2cam'])
        for i in range(N):
            intrinsic = np.array(results['cam2img'][i])
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            results['cam2img'][i] = viewpad
            img = Image.fromarray(np.uint8(imgs[i]))
            # augmentation (resize, crop, horizontal flip, rotate)
            # different view use different aug (BEV Det)
            img, ida_mat = self._img_transform(
                img,
                resize=resize,
                resize_dims=resize_dims,
                crop=crop,
                flip=flip,
                rotate=rotate,
            )
            new_imgs.append(np.array(img).astype(np.float32))
            results['cam2img'][
                i][:3, :3] = ida_mat @ results['cam2img'][i][:3, :3]

        results['img'] = new_imgs
        results['img_shape'] = new_imgs[0].shape[:2]

        return results

    def _get_rot(self, h):
        """
        从坐标系变换角度来看
        右手系, 1坐标系到2坐标系逆时针转h弧度有:
        R12=| cos(h) -sin(h) |
            | sin(h)  cos(h) |
        对于左手系(图像坐标系), 1坐标系到2坐标系逆时针转h弧度有:
        R12=| cos(h)  sin(h) |
            | -sin(h) cos(h) |

        但这里我们应该从一固定坐标系下点变换来看
        即图像点坐标沿着原点逆时针转h弧度, 有 
        p' = | ch   sh| p 
             | -sh  ch|
        """

        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def _img_transform(self, img, resize, resize_dims, crop, flip, rotate):
        """
        对图像的变换按以下顺序:
        1. 缩放
        2. 裁剪 crop = (lx, ly, rx, ry) 即在缩放后图上的左上角坐标(lx, ly)处裁剪到右下角坐标(rx, ry)处裁剪
        3. 左右翻转
        4. 旋转: 使用 PIL.Image.rotate(角度,非弧度), 逆时针为正, 以图像中心为旋转中心

        对应线性变换 p' = Mp, p是源像素坐标

        1. 缩放
        S = | s 0 0 |
            | 0 s 0 |
            | 0 0 1 |
        ps = Sp
        
        2. 裁减
        C = | 1 0 -lx |
            | 0 1 -ly |
            | 0 0   1 |
        pc = Cp

        3. 左右翻转
        F = | -1 0 rx-lx=crx |
            |  0 1 0 |
            |  0 0 1 |
        pf = Fp
        
        4. 旋转 分成三步: 整张图先平移到图像中心, 整张图旋转, 整张图再平移回原位置
        4.1 所有点位移
        t1 = | 1 0 -crx/2 |
             | 0 1 -cry/2 |
             | 0 0 1 |
        4.2 所有点逆时针旋转h弧度
        R2 = | ch  sh 0 |
             | -sh ch 0 |
             | 0    0 1 |
        4.3 所有点再平移回原位置
        t3 = | 1 0 crx/2 |
             | 0 1 cry/2 |
             | 0 0  1  |
        即 pr = (t3@R2@t1)p

        M = t3@R2@t1@F@C@S

        代码最后返回 变换后的img 和 M
        在PETR中, 使用了resize,crop,flip. 但没有使用旋转
        我们可以深入思考下,对于一对stereo, 使用resize,crop,flip, 依然可以保持几何关系一致, 但这里的旋转变换就会破坏
        """
        ida_rot = torch.eye(2)
        ida_tran = torch.zeros(2)
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        ida_rot *= resize
        ida_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            ida_rot = A.matmul(ida_rot)
            ida_tran = A.matmul(ida_tran) + b
        A = self._get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
        ida_mat = torch.eye(3)
        ida_mat[:2, :2] = ida_rot
        ida_mat[:2, 2] = ida_tran
        return img, ida_mat

    def _sample_augmentation(self):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.training:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.data_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate


@TRANSFORMS.register_module()
class GlobalRotScaleTransImage(BaseTransform):
    """Random resize, Crop and flip the image
    Args:
        size (tuple, optional): Fixed padding size.

    ------------------------------------------------------------
    gt在世界坐标系保持不变, 调整lidar外参(旋转, 缩放), 此时gt在cam的投影保持不变.
    1. 旋转lidar, 此时 gt(lidar系下坐标heading速度) 也要旋转
        随机生成h, Rno=| cos(h) -sin(h)  0|
                      | sin(h)  cos(h)  0|
                      | 0       0       1|
        对于gt的xy, 有 pn = Rno@po, pn 为新lidar系下的坐标, po 为原lidar系下的坐标
        对于gt的heading, 有 hn = h + ho, ho 为原lidar系下的heading, h 为旋转角度
        对于速度, 有 vn = Rno@vo, vo 为原lidar系下的速度
        原外参为 lidar2cam 是 Tclo
        新外参为 lidar2cam_new = Tclo @ Rno^{-1}
    
    2. 缩放lidar外参, 此时 gt(lidar系下的xylw速度)也要缩放
        随机生成s, Tno = | s 0 0 0 |
                        | 0 s 0 0 |
                        | 0 0 s 0 |
                        | 0 0 0 1 |
        对于gt的xyz, 有 pn = Tno@po, pn 为新lidar系下的坐标, po 为原lidar系下的坐标
        对于gt的lwh, 有 ln = s@lo, lo 为原lidar系下的lwh
        对于速度, 有 vn = s@vo, vo 为原lidar系下的速度
        原外参为 lidar2cam 是 Tclo
        新外参为 lidar2cam_new = Tclo @ Tno^{-1}
    """

    def __init__(
        self,
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        reverse_angle=False,
        training=True,
    ):

        self.rot_range = rot_range
        self.scale_ratio_range = scale_ratio_range
        self.translation_std = translation_std

        self.reverse_angle = reverse_angle
        self.training = training

    def transform(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        # random rotate
        rot_angle = np.random.uniform(*self.rot_range)

        self.rotate_bev_along_z(results, rot_angle)
        # if self.reverse_angle:
        #     rot_angle *= -1
        results['gt_bboxes_3d'].rotate(np.array(rot_angle))

        # random scale
        scale_ratio = np.random.uniform(*self.scale_ratio_range)
        self.scale_xyz(results, scale_ratio)
        results['gt_bboxes_3d'].scale(scale_ratio)

        # TODO: support translation
        # if not self.reverse_angle:
        #     gt_bboxes_3d = results['gt_bboxes_3d'].numpy()
        #     gt_bboxes_3d[:, 6] -= 2 * rot_angle
        #     results['gt_bboxes_3d'] = LiDARInstance3DBoxes(
        #         gt_bboxes_3d, box_dim=9)

        return results

    def rotate_bev_along_z(self, results, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array([[rot_cos, -rot_sin, 0, 0],
                                [rot_sin, rot_cos, 0, 0], [0, 0, 1, 0],
                                [0, 0, 0, 1]])
        rot_mat_inv = np.linalg.inv(rot_mat)
        num_view = len(results['lidar2cam'])
        for view in range(num_view):
            results['lidar2cam'][view] = (
                np.array(results['lidar2cam'][view])
                @ rot_mat_inv).astype(np.float32)

        return

    def scale_xyz(self, results, scale_ratio):
        rot_mat = np.array([
            [scale_ratio, 0, 0, 0],
            [0, scale_ratio, 0, 0],
            [0, 0, scale_ratio, 0],
            [0, 0, 0, 1],
        ])

        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results['lidar2cam'])
        for view in range(num_view):
            results['lidar2cam'][view] = (np.array(
                results['lidar2cam'][view]) @ rot_mat_inv).astype(np.float32)

        return
