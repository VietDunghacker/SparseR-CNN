#
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .config import add_sparsercnn_config
from .detector import SparseRCNN
from .dataset_mapper import SparseRCNNDatasetMapper

from .modeling.meta_arch.centernet_detector import CenterNetDetector
from .modeling.dense_heads.centernet import CenterNet
from .modeling.roi_heads.custom_roi_heads import CustomROIHeads, CustomCascadeROIHeads

from .modeling.backbone.fpn_p5 import build_p67_resnet_fpn_backbone
from .modeling.backbone.dla import build_dla_backbone
from .modeling.backbone.dlafpn import build_dla_fpn3_backbone
from .modeling.backbone.bifpn import build_resnet_bifpn_backbone
from .modeling.backbone.bifpn_fcos import build_fcos_resnet_bifpn_backbone
from .modeling.backbone.res2net import build_p67_res2net_fpn_backbone

from .data.datasets.objects365 import categories_v1
from .data.datasets.coco import _PREDEFINED_SPLITS_COCO