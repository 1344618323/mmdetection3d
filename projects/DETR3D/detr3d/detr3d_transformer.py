import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence,
                                         build_transformer_layer_sequence)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmengine.model import BaseModule, constant_init, xavier_init

from mmdet3d.registry import MODELS


def inverse_sigmoid(x, eps=1e-5):
    """Inverse function of sigmoid.

    Args:
        x (Tensor): The tensor to do the
            inverse.
        eps (float): EPS avoid numerical
            overflow. Defaults 1e-5.
    Returns:
        Tensor: The x has passed the inverse
            function of sigmoid, has same
            shape with input.
    """
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


@MODELS.register_module()
class Detr3DTransformer(BaseModule):
    """Implements the DETR3D transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        num_cams (int): Number of cameras in the dataset.
            Default: 6 in NuScenes Det.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.

    --------------------------------
    1. self.reference_points 是 (256, 3) 的线性层
    2. self.decoder 是 Detr3DTransformerDecoder 的 obj，其父类为 mmcv.cnn.bricks.transformer.TransformerLayerSequence
        该class源码见 /opt/conda/lib/python3.8/site-packages/mmcv/cnn/bricks/transformer.py
        Detr3DTransformerDecoder 只是重写了 TransformerLayerSequence 的forward方法，其他方法一样。
        TransformerLayerSequence 中会按配置构造多层(如6层) mmdet.models.layers.transformer.DetrTransformerDecoderLayer
            DetrTransformerDecoderLayer 源码见 /opt/conda/lib/python3.8/site-packages/mmdet/models/layers/transformer.py
            DetrTransformerDecoderLayer 的 父类是 mmcv.cnn.bricks.transformer.BaseTransformerLayer，这个类用于实现 一个transformer layer
                BaseTransformerLayer 源码 /opt/conda/lib/python3.8/site-packages/mmcv/cnn/bricks/transformer.py
                DetrTransformerDecoderLayer 构造时会先调用 BaseTransformerLayer 的构造函数, 
                所以先看下 BaseTransformerLayer 的构造函数做了啥:
                    1. 设置其内子块的run顺序, 如 ('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
                    2. 构造 self_attn 对应 MultiheadAttention, 源码见 /opt/conda/lib/python3.8/site-packages//cnn/bricks/transformer.py
                        这个类是 nn.MultiheadAttention的封装, 但要注意forward输出的是 x+multihead_attn(x)的结果
                    3. 构造 cross_attn 对应 Detr3DCrossAtten
                    4. 构建 norm 对应 LayerNorm, 源码见 /opt/conda/lib/python3.8/site-packages/mmcv/cnn/bricks/transformer.py
                        即 torch.nn.modules.normalization.LayerNorm
                    5. 构建 ffn 对应 FFN, 源码见 /opt/conda/lib/python3.8/site-packages/mmcv/cnn/bricks/transformer.py
                        具体实现为：linear(256, 512) -> ReLU -> dropout -> linear(512, 256) -> dropout
                        linear通道数，不一定是(256, 512)，看配置
                        forward输出的是 x+ffn(x)的结果

                    对于BaseTransformerLayer有必要再补充下，其中PE的使用方式与 attention is all you need 中不同：
                    在 attention is all you need 中, PE是先加到input上得到新input， 再串联 N 个 transformer layer；
                    而mmcv中的实现则在每一层 attention layer中反复注入query_pos, key_pos. 这是DETR的做法，见DETR论文table 3, 显示这样做能提点。

                    另外其残差使用方式有两种 pre_norm or post_norm
                    post_norm: x_{t+1} = LayerNorm(x_t + sublayer(x_t)) 原始 Transformer 采用, DETR3D 默认也是用这个
                    pre_norm: x_{t+1} = x_t + sublayer(LayerNorm(x_t)) 现代大模型中的标配，据说性能更好

                我们再看下BaseTransformerLayer.forward, 注意我们只看默认配置下的, 了解大意即可
                BaseTransformerLayer.forward(query, key, value, query_pos, key_pos, attn_masks, query_key_padding_mask, key_padding_mask, **kwargs):
                    1. query = self.attentions[0](query, query, query, None, query_pos, key_pos, ...)
                        即 masked multi-head self-attention and ADD(残差)
                    2. query = self.norms[0](query)
                        即 LayerNorm
                    3. query = self.attentions[1](query, key, value, None, query_pos, key_pos, ...)
                        即 cross-attention and ADD(残差)
                    4. query = self.norms[1](query)
                        即 LayerNorm
                    5. query = self.ffns[0](query)
                        即 FFN and ADD(残差)
                    6. query = self.norms[2](query)
                        即 LayerNorm
                    7. 返回query

        综上, 对于TransformerLayerSequence有, self.layers.__len__() == 6, 
        每个self.layers[i] 对应一个 DetrTransformerDecoderLayer, 其实就是一个 BaseTransformerLayer
            一个 BaseTransformerLayer 对应('self_attn', 'norm', 'cross_attn', 'norm', 'ffn', 'norm')
            即self.layers[i].attentions.__len__() == 2, 即 self_attn 和 cross_attn
            即self.layers[i].norms.__len__() == 3, 即 norm, norm, norm
            即self.layers[i].ffns.__len__() == 1, 即 ffn
    
    3. Detr3DTransformer.forward(mlvl_feats: 长度为[mlvl], 每个元素是 [B, N, C, H_lvl, W_lvl], 
            query_embed: [num_query, embed_dims*2] 是可学习参数, reg_branches=None, **kwargs):
        query_pos, query = 将query_embed拆成两段，每段都是 [num_query, embed_dims], 如 [900, 256], 并expand成 [B, num_query, embed_dims]
        reference_points = query_pos 经过 [num_query, 3] 的线性层 与 sigmoid 获取, 尺寸为 [B, num_query, 3]
        query_pos, query 都view成 [num_query, B, embed_dims]
        self.decoder(query, key=None, value=mlvl_feats, query_pos=query_pos, reference_points=reference_points, reg_branches=reg_branches, **kwargs)
            也就是 Detr3DTransformerDecoder.forward

        我们看下 Detr3DTransformerDecoder.forward:
            for lid, layer(即DetrTransformerDecoderLayer) in enumerate(self.layers):
                output = layer的forward(上一层的output这一层的query， reference_points, ...)
                如果配置了 with_box_refine, 会通过reg_branches[lid](output) 得到新的 reference_points
                    注意其实现逻辑，原reference_points是sigmoid后的
                    而通过reg_branches[lid](output) 回归的 reference_points偏移 是没有sigmoid 的
                    所以有 
                        new_reference_points = tmp[..., :2] + inverse_sigmoid(reference_points[..., :2])
                        new_reference_points[..., 2:3] = tmp[..., 4:5] + inverse_sigmoid(reference_points[..., 2:3])
                        new_reference_points = new_reference_points.sigmoid()
                        这几句代码
                    另外还要注意 reference_points = new_reference_points.detach() 也就说这些随着迭代变化的reference_points不会参与梯度回传
                如果没配置，则reference_points一直保持不变
            返回 output, reference_points
                若配置了self.return_intermediate，则尺寸分别是 [6, num_query, B, embed_dims] 和 [6, B, num_query, 3]
    
    Detr3DTransformer.forward 会返回 inter_states, init_reference_out, inter_references_out
        若配置了self.return_intermediate，则尺寸分别是 [6, num_query, B, embed_dims], [B, num_query, 3], [6, B, num_query, 3]
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 two_stage_num_proposals=300,
                 decoder=None,
                 **kwargs):
        super(Detr3DTransformer, self).__init__(**kwargs)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.decoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.two_stage_num_proposals = two_stage_num_proposals
        self.init_layers()

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.reference_points = nn.Linear(self.embed_dims, 3)

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention) or isinstance(
                    m, Detr3DCrossAtten):
                m.init_weight()
        xavier_init(self.reference_points, distribution='uniform', bias=0.)

    def forward(self, mlvl_feats, query_embed, reg_branches=None, **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                (B, N, C, H_lvl, W_lvl).
            query_embed (Tensor): The query positional and semantic embedding
                for decoder, with shape [num_query, c+c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                [bs, N, embed_dims, h, w]. It is unused here.
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape
                      (num_dec_layers, bs, num_query, embed_dims), else has
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference
                    points in decoder, has shape
                    (num_dec_layers, bs, num_query, embed_dims)
        
        --------------------------------
        关于这个expand，可以多了解一点，expand不是深拷贝，而是共用内存
        另外，不用担心反向传播的问题，可以把expeand理解成一个线性变换，如 [x] -> [x; x; x],
        其实就是 [1; 1; 1] * [x] = [x; x; x]
        """
        assert query_embed is not None
        bs = mlvl_feats[0].size(0)
        query_pos, query = torch.split(query_embed, self.embed_dims, dim=1)
        query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)  # [bs,num_q,c]
        query = query.unsqueeze(0).expand(bs, -1, -1)  # [bs,num_q,c]
        reference_points = self.reference_points(query_pos)
        reference_points = reference_points.sigmoid()
        init_reference_out = reference_points

        # decoder
        query = query.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)
        inter_states, inter_references = self.decoder(
            query=query,
            key=None,
            value=mlvl_feats,
            query_pos=query_pos,
            reference_points=reference_points,
            reg_branches=reg_branches,
            **kwargs)

        inter_references_out = inter_references
        return inter_states, init_reference_out, inter_references_out


@MODELS.register_module()
class Detr3DTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR3D transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default:
            `LN`.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(Detr3DTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

    def forward(self,
                query,
                *args,
                reference_points=None,
                reg_branches=None,
                **kwargs):
        """Forward function for `Detr3DTransformerDecoder`.
        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage,
                otherwise has shape self.reference_points =
                                        nn.Linear(self.embed_dims, 3)
            reg_branch: (obj:`nn.ModuleList`): Used for
                refining the regression results. Only would
                be passed when with_box_refine is True,
                otherwise would be passed a `None`.
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        output = query
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):  # iterative refinement
            reference_points_input = reference_points
            output = layer(
                output,
                *args,
                reference_points=reference_points_input,
                **kwargs)
            output = output.permute(1, 0, 2)
            if reg_branches is not None:
                tmp = reg_branches[lid](output)

                assert reference_points.shape[-1] == 3

                new_reference_points = torch.zeros_like(reference_points)
                new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(
                    reference_points[..., :2])
                new_reference_points[...,
                                     2:3] = tmp[..., 4:5] + inverse_sigmoid(
                                         reference_points[..., 2:3])
                new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            output = output.permute(1, 0, 2)
            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(
                intermediate_reference_points)

        return output, reference_points


@MODELS.register_module()
class Detr3DCrossAtten(BaseModule):
    """An attention module used in Detr3d.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(
        self,
        embed_dims=256,
        num_heads=8,
        num_levels=4,
        num_points=5,
        num_cams=6,
        im2col_step=64,
        pc_range=None,
        dropout=0.1,
        norm_cfg=None,
        init_cfg=None,
        batch_first=False,
    ):
        super(Detr3DCrossAtten, self).__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_cams = num_cams
        self.attention_weights = nn.Linear(embed_dims,
                                           num_cams * num_levels * num_points)

        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )
        self.batch_first = batch_first
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)

    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                reference_points=None,
                **kwargs):
        """Forward Function of Detr3DCrossAtten.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (List[Tensor]): Image features from
                different level. Each element has shape
                (B, N, C, H_lvl, W_lvl).
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            reference_points (Tensor): The normalized 3D reference
                points with shape (bs, num_query, 3)
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].

        --------------------------------
        torch.nan_to_num: nan->0, inf->max, -inf->min

        1. 输入 query: [B, num_query, embed_dims], reference_points: [B, num_query, 3]
            key直接赋值为query, 而query+=query_pos
        2. reference_points_3d, output, mask = feature_sampling(...) 返回reference_points在图像上对应的特征(双线性插值)
            reference_points_3d 就是 reference_points, shape [B, num_query, 3]
            output 是 [B, embed_dims, num_query, num_cam, 1, num_levels]
            mask 是 [B, 1, num_query, num_cam, 1, 1] 投影到图像内为true, 否则为false
        3. self.attention_weights 是 [embed_dims, num_cams * num_points * num_levels] 带bias的线性层
            attention_weights = self.attention_weights(query)
                结果会转成 [B, 1, num_query, num_cams, num_points, num_levels]
                并sigmoid得到 [B, 1, num_query, num_cams, num_points, num_levels]
            即通过线性层决定attention weights
            权重不是通过 Query 和 Key 计算余弦相似度得出的，而是直接由一个线性层对 Query 进行预测。
            它决定了该 Query 对不同相机 (num_cams)、不同采样点 (num_points) 和 不同特征层 (num_levels) 的重视程度    
        4. 加权融合: 即对一个query，将其在不同相机、不同采样点、不同特征层上的value加权融合
            attention_weights = attention_weights.sigmoid() * mask
            output = output * attention_weights
            output = output.sum(-1).sum(-1).sum(-1) # 依次对 levels, points, cams 求和
            尺寸为 [B, num_query, embed_dims], permute后为 [embed_dims, B, num_query]
        5. output = self.output_proj(output)
            线性投影层，其目的： 前面步骤通过 feature_sampling 从多张图像中采样的特征（经过求和融合后），其通道分布可能与 Query 原始的特征空间不一致。
                通过这个线性投影层，让模型学会如何“消化”这些从图像里捡出来的像素特征，将其转化为物体的高层语义
        6. pos_feat = self.position_encoder(inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
            inverse_sigmoid将[0,1]区间的reference_points_3d 反向映射回逻辑空间
            将这个 refpt 通过多层MLP生成对应的 3D 位置特征向量
            shape为 [num_query, B, embed_dims]
        7. self.dropout(output) + inp_residual + pos_feat
            融合 attention output 和 3D位置特征向量, 以及残差, 返回 [num_query, B, embed_dims]

        值得一提的是，DETR3D论文中提到的cross-attention，比这个还简单：query_{i+1} = output_i + query_i （残差连接）
        """
        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
        if query_pos is not None:
            query = query + query_pos

        query = query.permute(1, 0, 2)

        bs, num_query, _ = query.size()

        attention_weights = self.attention_weights(query).view(
            bs, 1, num_query, self.num_cams, self.num_points, self.num_levels)
        reference_points_3d, output, mask = feature_sampling(
            value, reference_points, self.pc_range, kwargs['img_metas'])
        output = torch.nan_to_num(output)
        mask = torch.nan_to_num(mask)
        attention_weights = attention_weights.sigmoid() * mask
        output = output * attention_weights
        output = output.sum(-1).sum(-1).sum(-1)
        output = output.permute(2, 0, 1)
        # (num_query, bs, embed_dims)
        output = self.output_proj(output)
        pos_feat = self.position_encoder(
            inverse_sigmoid(reference_points_3d)).permute(1, 0, 2)
        return self.dropout(output) + inp_residual + pos_feat


def feature_sampling(mlvl_feats,
                     ref_pt,
                     pc_range,
                     img_metas,
                     no_sampling=False):
    """ sample multi-level features by projecting 3D reference points
            to 2D image
        Args:
            mlvl_feats (List[Tensor]): Image features from
                different level. Each element has shape
                (B, N, C, H_lvl, W_lvl).
            ref_pt (Tensor): The normalized 3D reference
                points with shape (bs, num_query, 3)
            pc_range: perception range of the detector
            img_metas (list[dict]): Meta information of multiple inputs
                in a batch, containing `lidar2img`.
            no_sampling (bool): If set 'True', the function will return
                2D projected points and mask only.
        Returns:
            ref_pt_3d (Tensor): A copy of original ref_pt
            sampled_feats (Tensor): sampled features with shape \
                (B C num_q N 1 fpn_lvl)
            mask (Tensor): Determine whether the reference point \
                has projected outsied of images, with shape \
                (B 1 num_q N 1 1)

    --------------------------------
    ref_pt表示lidar坐标系下参考点，投影到图像上，并通过bilinear插值得到特征
    
    两个细节：
    1. input ref_pt是通过sigmoid得到的[0, 1]范围，所以要通过pc_range转换到物理坐标下, 再做投影
    2. F.grid_sample 会将输入坐标从xy[-1~1, -1~1]映射到[0~H, 0~W]范围

    返回
    ref_pt_3d: 就是input ref_pt, 值范围是[0,1]， shape [B, num_query, 3]
    sampled_feats: 插值得到的特征， shape [B, embed_dims, num_query, num_cam, 1, mlvl]
    mask: true表示哪些点投影到了图像内， shape [B, 1, num_query, num_cam, 1, 1]
    """
    lidar2img = [meta['lidar2img'] for meta in img_metas]
    lidar2img = np.asarray(lidar2img)
    lidar2img = ref_pt.new_tensor(lidar2img)
    ref_pt = ref_pt.clone()
    ref_pt_3d = ref_pt.clone()

    B, num_query = ref_pt.size()[:2]
    num_cam = lidar2img.size(1)
    eps = 1e-5

    ref_pt[..., 0:1] = \
        ref_pt[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]  # x
    ref_pt[..., 1:2] = \
        ref_pt[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]  # y
    ref_pt[..., 2:3] = \
        ref_pt[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]  # z

    # (B num_q 3) -> (B num_q 4) -> (B 1 num_q 4) -> (B num_cam num_q 4 1)
    ref_pt = torch.cat((ref_pt, torch.ones_like(ref_pt[..., :1])), -1)
    ref_pt = ref_pt.view(B, 1, num_query, 4)
    ref_pt = ref_pt.repeat(1, num_cam, 1, 1).unsqueeze(-1)
    # (B num_cam 4 4) -> (B num_cam num_q 4 4)
    lidar2img = lidar2img.view(B, num_cam, 1, 4, 4)\
                         .repeat(1, 1, num_query, 1, 1)
    # (... 4 4) * (... 4 1) -> (B num_cam num_q 4)
    pt_cam = torch.matmul(lidar2img, ref_pt).squeeze(-1)

    # (B num_cam num_q)
    z = pt_cam[..., 2:3]
    eps = eps * torch.ones_like(z)
    mask = (z > eps)
    pt_cam = pt_cam[..., 0:2] / torch.maximum(z, eps)  # prevent zero-division
    # padded nuscene image: 928*1600
    (h, w) = img_metas[0]['pad_shape']
    pt_cam[..., 0] /= w
    pt_cam[..., 1] /= h
    # else:
    # (h,w,_) = img_metas[0]['ori_shape'][0]          # waymo image
    # pt_cam[..., 0] /= w # cam0~2: 1280*1920
    # pt_cam[..., 1] /= h # cam3~4: 886 *1920 padded to 1280*1920
    # mask[:, 3:5, :] &= (pt_cam[:, 3:5, :, 1:2] < 0.7) # filter pt_cam_y > 886

    mask = (
        mask & (pt_cam[..., 0:1] > 0.0)
        & (pt_cam[..., 0:1] < 1.0)
        & (pt_cam[..., 1:2] > 0.0)
        & (pt_cam[..., 1:2] < 1.0))

    if no_sampling:
        return pt_cam, mask

    # (B num_cam num_q) -> (B 1 num_q num_cam 1 1)
    mask = mask.view(B, num_cam, 1, num_query, 1, 1).permute(0, 2, 3, 1, 4, 5)
    mask = torch.nan_to_num(mask)

    pt_cam = (pt_cam - 0.5) * 2  # [0,1] to [-1,1] to do grid_sample
    sampled_feats = []
    for lvl, feat in enumerate(mlvl_feats):
        B, N, C, H, W = feat.size()
        feat = feat.view(B * N, C, H, W)
        pt_cam_lvl = pt_cam.view(B * N, num_query, 1, 2)
        sampled_feat = F.grid_sample(feat, pt_cam_lvl)
        # (B num_cam C num_query 1) -> List of (B C num_q num_cam 1)
        sampled_feat = sampled_feat.view(B, N, C, num_query, 1)
        sampled_feat = sampled_feat.permute(0, 2, 3, 1, 4)
        sampled_feats.append(sampled_feat)

    sampled_feats = torch.stack(sampled_feats, -1)
    # (B C num_q num_cam fpn_lvl)
    sampled_feats = \
        sampled_feats.view(B, C, num_query, num_cam, 1, len(mlvl_feats))
    return ref_pt_3d, sampled_feats, mask
