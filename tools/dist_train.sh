#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

export NCCL_P2P_DISABLE=1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}

: <<'COMMENT'
4卡训练pointpillars
./tools/dist_train.sh configs/pointpillars/configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py 4 \
    --work-dir /mnt/intel/jupyterhub/xinning/mmdet3d_work_dir/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d \

测试pointpillars
./tools/dist_test.sh /mnt/intel/jupyterhub/xinning/mmdet3d_work_dir/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py \
    /mnt/intel/jupyterhub/xinning/mmdet3d_work_dir/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d/epoch_24.pth 4

测试pointpillars
./tools/dist_test.sh /mnt/intel/jupyterhub/xinning/mmdet3d_work_dir/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py \
    /mnt/intel/jupyterhub/xinning/mmdet3d_pretrain/pointpillars/hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth 8 \
    --cfg-options test_evaluator.jsonfile_prefix=/mnt/intel/jupyterhub/xinning/mmdet3d_work_dir/pretrained_model_eval_results

训练centerpoint
./tools/dist_train.sh configs/centerpoint/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py 8 \
    --work-dir /mnt/intel/jupyterhub/xinning/mmdet3d_work_dir/centerpoint_voxel0075_second_secfpn_head-dcn-circlenms_8xb4-cyclic-20e_nus-3d

训练DETR3D
当前image的conda环境不兼容detr3d, 需执行
1. pip install "mmdet<=3.0.0rc5"
2. pip install "mmcv>=2.0.0rc0, <2.1.0"
3. 修改/opt/conda/lib/python3.8/site-packages/mmdet/models/data_preprocessors/data_preprocessor.py

@MODELS.register_module()
class DetDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for detection tasks.

    Comparing with the :class:`mmengine.ImgDataPreprocessor`,

    1. It supports batch augmentations.
    2. It will additionally append batch_input_shape and pad_shape
    to data_samples considering the object detection task.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Do batch augmentations during training.

    Args:
        mean (Sequence[Number], optional): The pixel mean of R, G, B channels.
            Defaults to None.
        std (Sequence[Number], optional): The pixel standard deviation of
            R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (Number): The padded pixel value. Defaults to 0.
        pad_mask (bool): Whether to pad instance masks. Defaults to False.
        mask_pad_value (int): The padded pixel value for instance masks.
            Defaults to 0.
        pad_seg (bool): Whether to pad semantic segmentation maps.
            Defaults to False.
        seg_pad_value (int): The padded pixel value for semantic
            segmentation maps. Defaults to 255.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to RGB.
            Defaults to False.
        boxtype2tensor (bool): Whether to keep the ``BaseBoxes`` type of
            bboxes data or not. Defaults to False.
        batch_augments (list[dict], optional): Batch-level augmentations
    """

    def __init__(self,
                 mean: Sequence[Number] = None,
                 std: Sequence[Number] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 pad_mask: bool = False,
                 mask_pad_value: int = 0,
                 pad_seg: bool = False,
                 seg_pad_value: int = 255,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 boxtype2tensor: bool = True,
                 non_blocking: bool = False, # 新增行
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking) # 新增行


bash tools/dist_train.sh projects/DETR3D/configs/detr3d_r101_gridmask_cbgs.py 6 --cfg-options load_from=/mnt/intel/jupyterhub/xinning/mmdet3d_pretrain/detr3d/fcos3d.pth \
    --work-dir /mnt/intel/jupyterhub/xinning/mmdet3d_work_dir/detr3d_r101_gridmask_cbgs
COMMENT