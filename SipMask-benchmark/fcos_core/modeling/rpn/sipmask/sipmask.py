import math
import torch
import torch.nn.functional as F
from torch import nn

from .inference import make_sipmask_postprocessor
from .loss import make_sipmask_loss_evaluator

from fcos_core.layers import Scale
from fcos_core.layers import DFConv2d
from fcos_core.layers import DeformConv


class FeatureAlign(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4):
        super(FeatureAlign, self).__init__()
        offset_channels = kernel_size * kernel_size * 2
        self.conv_offset = nn.Conv2d(4,
                                     deformable_groups * offset_channels,
                                     1,
                                     bias=False)
        self.conv_adaption = DeformConv(in_channels,
                                        out_channels,
                                        kernel_size=kernel_size,
                                        padding=(kernel_size - 1) // 2,
                                        deformable_groups=deformable_groups,bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.GroupNorm(32, in_channels)



    def init_weights(self,bias_value=0):

        torch.nn.init.normal_(self.conv_offset.weight, std=0.0)
        torch.nn.init.normal_(self.conv_adaption.weight, std=0.01)
        torch.nn.init.constant_(self.conv_adaption.bias, bias_value)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.norm(self.conv_adaption(x, offset)))
        return x


class SipMaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(SipMaskHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        num_classes = cfg.MODEL.SIPMASK.NUM_CLASSES - 1
        self.fpn_strides = cfg.MODEL.SIPMASK.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.SIPMASK.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.SIPMASK.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.SIPMASK.USE_DCN_IN_TOWER

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.SIPMASK.NUM_CONVS-1):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.SIPMASK.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            cls_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

        for i in range(cfg.MODEL.SIPMASK.NUM_CONVS):
            if self.use_dcn_in_tower and \
                    i == cfg.MODEL.SIPMASK.NUM_CONVS - 1:
                conv_func = DFConv2d
            else:
                conv_func = nn.Conv2d

            bbox_tower.append(
                conv_func(
                    in_channels,
                    in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=True
                )
            )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())
        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )

        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        
        self.nc = 32
        ###########instance##############
        self.feat_align = FeatureAlign(in_channels, in_channels, 3)
        self.sip_cof = nn.Conv2d(in_channels, self.nc*4, 3, padding=1)

        self.sip_mask_lat = nn.Conv2d(512, self.nc, 3, padding=1)
        self.sip_mask_lat0 = nn.Conv2d(768, 512, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.bbox_pred, self.cls_logits,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.SIPMASK.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])
        self.feat_align.init_weights()

        
    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        cof_preds = []
        count = 0
        feat_masks = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))

            cls_tower1 = self.feat_align(cls_tower, bbox_pred)
            logits.append(self.cls_logits(cls_tower1))

            ########COFFECIENTS###############
            cof_pred = self.sip_cof(cls_tower1)
            cof_preds.append(cof_pred)


            ############contextual#######################
            if count < 3:
                feat_up = F.interpolate(box_tower, scale_factor=(2 ** count), mode='bilinear', align_corners=False)
                feat_masks.append(feat_up)
            count = count + 1

        # ################contextual enhanced##################
        feat_masks = torch.cat(feat_masks, dim=1)
        
        feat_masks = self.relu(self.sip_mask_lat(self.relu(self.sip_mask_lat0(feat_masks))))
        feat_masks = F.interpolate(feat_masks, scale_factor=4, mode='bilinear', align_corners=False)

        return logits, bbox_reg, centerness, cof_preds, feat_masks


class SipMaskModule(torch.nn.Module):
    """
    Module for SipMask computation. Takes feature maps from the backbone and
    SipMask outputs and losses. Only Test on FPN now.
    """

    def __init__(self, cfg, in_channels):
        super(SipMaskModule, self).__init__()

        head = SipMaskHead(cfg, in_channels)

        box_selector_test = make_sipmask_postprocessor(cfg)

        loss_evaluator = make_sipmask_loss_evaluator(cfg)
        self.head = head
        self.box_selector_test = box_selector_test
        self.loss_evaluator = loss_evaluator
        self.fpn_strides = cfg.MODEL.SIPMASK.FPN_STRIDES
        
    def forward(self, images, features, img_metas=None, targets=None):
        """
        Arguments:
            images (ImageList): images for which we want to compute the predictions
            features (list[Tensor]): features computed from the images that are
                used for computing the predictions. Each tensor in the list
                correspond to different feature levels
            targets (list[BoxList): ground-truth boxes present in the image (optional)

        Returns:
            boxes (list[BoxList]): the predicted boxes from the RPN, one BoxList per
                image.
            losses (dict[Tensor]): the losses for the model during training. During
                testing, it is an empty dict.
        """
        box_cls, box_regression, centerness, box_cof, box_mask = self.head(features)
        locations = self.compute_locations(features)
 
        if self.training:
            return self._forward_train(
                locations, box_cls, 
                box_regression, 
                centerness, box_cof, box_mask, targets
            )
        else:
            return self._forward_test(
                locations, box_cls, box_regression, 
                centerness, box_cof, box_mask, images.image_sizes, img_metas
            )

    def _forward_train(self, locations, box_cls, box_regression, centerness, box_cof, feat_mask, targets):
        loss_box_cls, loss_box_reg, loss_centerness, loss_mask = self.loss_evaluator(
            locations, box_cls, box_regression, centerness, box_cof, feat_mask, targets
        )
        losses = {
            "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness,
            "loss_mask": loss_mask
        }
        return None, losses

    def _forward_test(self, locations, box_cls, box_regression, centerness, box_cof, box_mask, image_sizes, img_metas):
        boxes = self.box_selector_test(
            locations, box_cls, box_regression, 
            centerness, box_cof, box_mask, image_sizes, img_metas
        )
        return boxes, {}

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations

def build_sipmask(cfg, in_channels):
    return SipMaskModule(cfg, in_channels)
