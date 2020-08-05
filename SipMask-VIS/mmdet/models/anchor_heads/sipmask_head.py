import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import distance2bbox, bbox_overlaps, force_fp32, multi_apply, multiclass_nms, multiclass_nms_idx
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob, ConvModule, Scale
from mmdet.ops import DeformConv, CropSplit, CropSplitGt
from ..losses import cross_entropy, accuracy
import torch.nn.functional as F
import pycocotools.mask as mask_util
import numpy as np

INF = 1e8


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(((boxes[:, 2:] + boxes[:, :2]) / 2,  # cx, cy
                      boxes[:, 2:] - boxes[:, :2]), 1)  # w, h


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
                                        deformable_groups=deformable_groups)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.GroupNorm(32, in_channels)

    def init_weights(self, bias_value=0):
        torch.nn.init.normal_(self.conv_offset.weight, std=0.0)
        torch.nn.init.normal_(self.conv_adaption.weight, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        x = self.relu(self.norm(self.conv_adaption(x, offset)))
        return x


def crop_split(masks00, masks01, masks10, masks11, boxes, masksG=None):
    """
    "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    Vectorized by Chong (thanks Chong).

    Args:
        - masks should be a size [h, w, n] tensor of masks
        - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    """

    h, w, n = masks00.size()
    rows = torch.arange(w, device=masks00.device, dtype=boxes.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks00.device, dtype=boxes.dtype).view(-1, 1, 1).expand(h, w, n)

    x1, x2 = boxes[:, 0], boxes[:, 2]  # sanitize_coordinates(boxes[:, 0], boxes[:, 2], w, padding, cast=False)
    y1, y2 = boxes[:, 1], boxes[:, 3]  # sanitize_coordinates(boxes[:, 1], boxes[:, 3], h, padding, cast=False)
    xc = (x1 + x2) / 2
    yc = (y1 + y2) / 2
    x1 = torch.clamp(x1, min=0, max=w - 1)
    y1 = torch.clamp(y1, min=0, max=h - 1)
    x2 = torch.clamp(x2, min=0, max=w - 1)
    y2 = torch.clamp(y2, min=0, max=h - 1)
    xc = torch.clamp(xc, min=0, max=w - 1)
    yc = torch.clamp(yc, min=0, max=h - 1)

    ##x1,y1,xc,yc
    crop_mask = (rows >= x1.view(1, 1, -1)) & (rows < xc.view(1, 1, -1)) & (cols >= y1.view(1, 1, -1)) & (
                cols < yc.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()

    masks00 = masks00 * crop_mask

    ##xc,y1,x2,yc
    crop_mask = (rows >= xc.view(1, 1, -1)) & (rows < x2.view(1, 1, -1)) & (cols >= y1.view(1, 1, -1)) & (
                cols < yc.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks01 = masks01 * crop_mask

    crop_mask = (rows >= x1.view(1, 1, -1)) & (rows < xc.view(1, 1, -1)) & (cols >= yc.view(1, 1, -1)) & (
                cols < y2.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks10 = masks10 * crop_mask

    crop_mask = (rows >= xc.view(1, 1, -1)) & (rows < x2.view(1, 1, -1)) & (cols >= yc.view(1, 1, -1)) & (
                cols < y2.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks11 = masks11 * crop_mask

    masks = masks00 + masks01 + masks10 + masks11

    ########whole
    if masksG is not None:
        crop_mask = (rows >= x1.view(1, 1, -1)) & (rows < x2.view(1, 1, -1)) & (cols >= y1.view(1, 1, -1)) & (
                    cols < y2.view(1, 1, -1))
        crop_mask = crop_mask.float()

        masksG = masksG * crop_mask
        return masks, masksG

    return masks


@HEADS.register_module
class SipMaskHead(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(4, 8, 16, 32, 64),
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):
        super(SipMaskHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes - 1
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.regress_ranges = regress_ranges
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.fp16_enabled = False
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.match_coeff = [1.0, 2.0, 10]

        self.loss_track = build_loss(dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0))
        self.prev_roi_feats = None
        self.prev_bboxes = None
        self.prev_det_labels = None
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs - 1):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.fcos_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.fcos_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.fcos_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)

        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

        self.nc = 32
        ###########instance##############
        self.feat_align = FeatureAlign(self.feat_channels, self.feat_channels, 3)
        self.sip_cof = nn.Conv2d(self.feat_channels, self.nc * 4, 3, padding=1)

        self.sip_mask_lat = nn.Conv2d(512, self.nc, 3, padding=1)
        self.sip_mask_lat0 = nn.Conv2d(768, 512, 1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.crop_cuda = CropSplit(2)
        self.crop_gt_cuda = CropSplitGt(2)

        self.track_convs = nn.ModuleList()
        for i in range(self.stacked_convs - 1):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.track_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None))
        self.sipmask_track = nn.Conv2d(self.feat_channels * 3, 512, 1, padding=0)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

        normal_init(self.sip_cof, std=0.01)
        normal_init(self.sip_mask_lat, std=0.01)
        normal_init(self.sip_mask_lat0, std=0.01)
        self.feat_align.init_weights()

        for m in self.track_convs:
            normal_init(m.conv, std=0.01)

    def forward(self, feats, feats_x, flag_train=True):
        # return multi_apply(self.forward_single, feats, self.scales)
        cls_scores = []
        bbox_preds = []
        centernesses = []
        cof_preds = []
        feat_masks = []
        track_feats = []
        track_feats_ref = []
        count = 0
        for x, x_f, scale, stride in zip(feats, feats_x, self.scales, self.strides):
            cls_feat = x
            reg_feat = x
            track_feat = x
            track_feat_f = x_f

            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            if count < 3:
                for track_layer in self.track_convs:
                    track_feat = track_layer(track_feat)
                track_feat = F.interpolate(track_feat, scale_factor=(2 ** count), mode='bilinear', align_corners=False)
                track_feats.append(track_feat)
                if flag_train:
                    for track_layer in self.track_convs:
                        track_feat_f = track_layer(track_feat_f)
                    track_feat_f = F.interpolate(track_feat_f, scale_factor=(2 ** count), mode='bilinear',
                                                 align_corners=False)
                    track_feats_ref.append(track_feat_f)
            # scale the bbox_pred of different level
            # float to avoid overflow when enabling FP16
            bbox_pred = scale(self.fcos_reg(reg_feat))

            cls_feat = self.feat_align(cls_feat, bbox_pred)
            cls_score = self.fcos_cls(cls_feat)
            centerness = self.fcos_centerness(reg_feat)
            centernesses.append(centerness)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred.float() * stride)

            ########COFFECIENTS###############
            cof_pred = self.sip_cof(cls_feat)
            cof_preds.append(cof_pred)

            ############contextual#######################
            if count < 3:
                feat_up = F.interpolate(reg_feat, scale_factor=(2 ** count), mode='bilinear', align_corners=False)
                feat_masks.append(feat_up)
            count = count + 1
        # ################contextual enhanced##################
        feat_masks = torch.cat(feat_masks, dim=1)
        feat_masks = self.relu(self.sip_mask_lat(self.relu(self.sip_mask_lat0(feat_masks))))
        feat_masks = F.interpolate(feat_masks, scale_factor=4, mode='bilinear', align_corners=False)

        track_feats = torch.cat(track_feats, dim=1)
        track_feats = self.sipmask_track(track_feats)
        if flag_train:
            track_feats_ref = torch.cat(track_feats_ref, dim=1)
            track_feats_ref = self.sipmask_track(track_feats_ref)
            return cls_scores, bbox_preds, centernesses, cof_preds, feat_masks, track_feats, track_feats_ref
        else:
            return cls_scores, bbox_preds, centernesses, cof_preds, feat_masks, track_feats, track_feats

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             cof_preds,
             feat_masks,
             track_feats,
             track_feats_ref,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None,
             gt_masks_list=None,
             ref_bboxes_list=None,
             gt_pids_list=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        labels, bbox_targets, label_list, bbox_targets_list, gt_inds = self.fcos_target(all_level_points,
                                                                                        gt_bboxes, gt_labels)

        # decode detection and groundtruth
        det_bboxes = []
        det_targets = []
        num_levels = len(bbox_preds)

        for img_id in range(len(img_metas)):
            bbox_pred_list = [
                bbox_preds[i][img_id].permute(1, 2, 0).reshape(-1, 4).detach() for i in range(num_levels)
            ]
            bbox_target_list = bbox_targets_list[img_id]

            bboxes = []
            targets = []
            for i in range(len(bbox_pred_list)):
                bbox_pred = bbox_pred_list[i]
                bbox_target = bbox_target_list[i]
                points = all_level_points[i]
                bboxes.append(distance2bbox(points, bbox_pred))
                targets.append(distance2bbox(points, bbox_target))

            bboxes = torch.cat(bboxes, dim=0)
            targets = torch.cat(targets, dim=0)

            det_bboxes.append(bboxes)
            det_targets.append(targets)
        gt_masks = []
        for i in range(len(gt_labels)):
            gt_label = gt_labels[i]
            gt_masks.append(
                torch.from_numpy(np.array(gt_masks_list[i][:gt_label.shape[0]], dtype=np.float32)).to(gt_label.device))

        num_imgs = cls_scores[0].size(0)
        # flatten cls_scores, bbox_preds and centerness
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        pos_inds = flatten_labels.nonzero().reshape(-1)
        num_pos = len(pos_inds)
        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels,
            avg_factor=num_pos + num_imgs)  # avoid num_pos is 0

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]

        if num_pos > 0:
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_centerness_targets = self.centerness_target(pos_bbox_targets)
            pos_points = flatten_points[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets)
            # centerness weighted iou loss
            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_target_preds,
                weight=pos_centerness_targets,
                avg_factor=pos_centerness_targets.sum())
            loss_centerness = self.loss_centerness(pos_centerness,
                                                   pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()

        ##########mask loss#################
        flatten_cls_scores1 = [
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs, -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores1 = torch.cat(flatten_cls_scores1, dim=1)

        flatten_cof_preds = [
            cof_pred.permute(0, 2, 3, 1).reshape(cof_pred.shape[0], -1, 32 * 4)
            for cof_pred in cof_preds
        ]

        loss_mask = 0
        loss_match = 0
        match_acc = 0
        n_total = 0
        flatten_cof_preds = torch.cat(flatten_cof_preds, dim=1)

        for i in range(num_imgs):
            labels = torch.cat([labels_level.flatten() for labels_level in label_list[i]])
            bbox_dt = det_bboxes[i] / 2
            bbox_dt = bbox_dt.detach()
            pos_inds = (labels > 0).nonzero().view(-1)
            cof_pred = flatten_cof_preds[i][pos_inds]
            img_mask = feat_masks[i]
            mask_h = img_mask.shape[1]
            mask_w = img_mask.shape[2]
            idx_gt = gt_inds[i]
            bbox_dt = bbox_dt[pos_inds, :4]

            area = (bbox_dt[:, 2] - bbox_dt[:, 0]) * (bbox_dt[:, 3] - bbox_dt[:, 1])
            bbox_dt = bbox_dt[area > 1.0, :]
            idx_gt = idx_gt[area > 1.0]
            cof_pred = cof_pred[area > 1.0]
            if bbox_dt.shape[0] == 1.0:
                loss_mask += area.sum()*0
                continue

            bbox_gt = gt_bboxes[i]
            cls_score = flatten_cls_scores1[i, pos_inds, labels[pos_inds] - 1].sigmoid().detach()
            cls_score = cls_score[area > 1.0]
            ious = bbox_overlaps(bbox_gt[idx_gt] / 2, bbox_dt, is_aligned=True)
            weighting = cls_score * ious
            weighting = weighting / (torch.sum(weighting) + 0.0001) * len(weighting)

            ###################track####################
            bboxes = ref_bboxes_list[i]
            amplitude = 0.05
            random_offsets = bboxes.new_empty(bboxes.shape[0], 4).uniform_(
                -amplitude, amplitude)
            # before jittering
            cxcy = (bboxes[:, 2:4] + bboxes[:, :2]) / 2
            wh = (bboxes[:, 2:4] - bboxes[:, :2]).abs()
            # after jittering
            new_cxcy = cxcy + wh * random_offsets[:, :2]
            new_wh = wh * (1 + random_offsets[:, 2:])
            # xywh to xyxy
            new_x1y1 = (new_cxcy - new_wh / 2)
            new_x2y2 = (new_cxcy + new_wh / 2)
            new_bboxes = torch.cat([new_x1y1, new_x2y2], dim=1)
            # clip bboxes
            # print(bbox_dt.shape)
            track_feat_i = self.extract_box_feature_center_single(track_feats[i], bbox_dt * 2)
            track_box_ref = self.extract_box_feature_center_single(track_feats_ref[i], new_bboxes)

            gt_pids = gt_pids_list[i]
            cur_ids = gt_pids[idx_gt]
            prod = torch.mm(track_feat_i, torch.transpose(track_box_ref, 0, 1))
            m = prod.size(0)
            dummy = torch.zeros(m, 1, device=torch.cuda.current_device())

            prod_ext = torch.cat([dummy, prod], dim=1)
            loss_match += cross_entropy(prod_ext, cur_ids)
            n_total += len(idx_gt)
            match_acc += accuracy(prod_ext, cur_ids) * len(idx_gt)

            gt_mask = F.interpolate(gt_masks[i].unsqueeze(0), scale_factor=0.5, mode='bilinear',
                                    align_corners=False).squeeze(0)

            shape = np.minimum(feat_masks[i].shape, gt_mask.shape)
            gt_mask_new = gt_mask.new_zeros(gt_mask.shape[0], mask_h, mask_w)
            gt_mask_new[:gt_mask.shape[0], :shape[1], :shape[2]] = gt_mask[:gt_mask.shape[0], :shape[1], :shape[2]]
            gt_mask_new = gt_mask_new.gt(0.5).float()

            gt_mask_new = torch.index_select(gt_mask_new, 0, idx_gt).permute(1, 2, 0).contiguous()

            #######spp###########################
            img_mask1 = img_mask.permute(1, 2, 0)
            pos_masks00 = torch.sigmoid(img_mask1 @ cof_pred[:, 0:32].t())
            pos_masks01 = torch.sigmoid(img_mask1 @ cof_pred[:, 32:64].t())
            pos_masks10 = torch.sigmoid(img_mask1 @ cof_pred[:, 64:96].t())
            pos_masks11 = torch.sigmoid(img_mask1 @ cof_pred[:, 96:128].t())
            pred_masks = torch.stack([pos_masks00, pos_masks01, pos_masks10, pos_masks11], dim=0)
            pred_masks = self.crop_cuda(pred_masks, bbox_dt)
            gt_mask_crop = self.crop_gt_cuda(gt_mask_new, bbox_dt)
            # pred_masks, gt_mask_crop = crop_split(pos_masks00, pos_masks01, pos_masks10, pos_masks11, bbox_dt,
            #                                       gt_mask_new)
            pre_loss = F.binary_cross_entropy(pred_masks, gt_mask_crop, reduction='none')

            pos_get_csize = center_size(bbox_dt)
            gt_box_width = pos_get_csize[:, 2]
            gt_box_height = pos_get_csize[:, 3]
            pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height / pos_get_csize.shape[0]
            loss_mask += torch.sum(pre_loss * weighting.detach())

        loss_mask = loss_mask / num_imgs
        loss_match = loss_match / num_imgs
        match_acc = match_acc / n_total
        if loss_mask == 0:
            loss_mask = bbox_dt[:, 0].sum()*0

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_mask=loss_mask,
            loss_match=loss_match,
            match_acc=match_acc)

    def compute_comp_scores(self, match_ll, bbox_scores, bbox_ious, label_delta, add_bbox_dummy=False):
        # compute comprehensive matching score based on matchig likelihood,
        # bbox confidence, and ious
        if add_bbox_dummy:
            bbox_iou_dummy = torch.ones(bbox_ious.size(0), 1,
                                        device=torch.cuda.current_device()) * 0
            bbox_ious = torch.cat((bbox_iou_dummy, bbox_ious), dim=1)
            label_dummy = torch.ones(bbox_ious.size(0), 1,
                                     device=torch.cuda.current_device())
            label_delta = torch.cat((label_dummy, label_delta), dim=1)

        if self.match_coeff is None:
            return match_ll
        else:
            # match coeff needs to be length of 3
            assert (len(self.match_coeff) == 3)
            return match_ll + self.match_coeff[0] * \
                   torch.log(bbox_scores) + self.match_coeff[1] * bbox_ious \
                   + self.match_coeff[2] * label_delta

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   cof_preds,
                   feat_masks,
                   track_feats,
                   track_feats_ref,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            cof_pred_list = [
                cof_preds[i][img_id].detach() for i in range(num_levels)
            ]
            feat_mask_list = feat_masks[img_id]
            track_feat_list = track_feats[img_id]
            is_first = img_metas[img_id]['is_first']

            img_shape = img_metas[img_id]['img_shape']
            ori_shape = img_metas[img_id]['ori_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list, cof_pred_list, feat_mask_list,
                                                mlvl_points, img_shape, ori_shape,
                                                scale_factor, cfg, rescale)
            if det_bboxes[0].shape[0] == 0:
                cls_segms = [[] for _ in range(self.num_classes - 1)]
                result_list.append([det_bboxes[0], det_bboxes[1], cls_segms, []])
                return result_list
            res_det_bboxes = det_bboxes[0] + 0.0
            if rescale:
                res_det_bboxes[:, :4] *= scale_factor

            det_roi_feats = self.extract_box_feature_center_single(track_feat_list, res_det_bboxes[:, :4])

            # recompute bbox match feature
            det_labels = det_bboxes[1]
            if is_first or (not is_first and self.prev_bboxes is None):
                det_obj_ids = np.arange(res_det_bboxes.size(0))
                # save bbox and features for later matching
                self.prev_bboxes = det_bboxes[0]
                self.prev_roi_feats = det_roi_feats
                self.prev_det_labels = det_labels
            else:

                assert self.prev_roi_feats is not None
                # only support one image at a time
                prod = torch.mm(det_roi_feats, torch.transpose(self.prev_roi_feats, 0, 1))
                m = prod.size(0)
                dummy = torch.zeros(m, 1, device=torch.cuda.current_device())
                match_score = torch.cat([dummy, prod], dim=1)
                match_logprob = torch.nn.functional.log_softmax(match_score, dim=1)
                label_delta = (self.prev_det_labels == det_labels.view(-1, 1)).float()
                bbox_ious = bbox_overlaps(det_bboxes[0][:, :4], self.prev_bboxes[:, :4])
                # compute comprehensive score
                comp_scores = self.compute_comp_scores(match_logprob,
                                                       det_bboxes[0][:, 4].view(-1, 1),
                                                       bbox_ious,
                                                       label_delta,
                                                       add_bbox_dummy=True)
                match_likelihood, match_ids = torch.max(comp_scores, dim=1)
                # translate match_ids to det_obj_ids, assign new id to new objects
                # update tracking features/bboxes of exisiting object,
                # add tracking features/bboxes of new object
                match_ids = match_ids.cpu().numpy().astype(np.int32)
                det_obj_ids = np.ones((match_ids.shape[0]), dtype=np.int32) * (-1)
                best_match_scores = np.ones((self.prev_bboxes.size(0))) * (-100)
                for idx, match_id in enumerate(match_ids):
                    if match_id == 0:
                        # add new object
                        det_obj_ids[idx] = self.prev_roi_feats.size(0)
                        self.prev_roi_feats = torch.cat((self.prev_roi_feats, det_roi_feats[idx][None]), dim=0)
                        self.prev_bboxes = torch.cat((self.prev_bboxes, det_bboxes[0][idx][None]), dim=0)
                        self.prev_det_labels = torch.cat((self.prev_det_labels, det_labels[idx][None]), dim=0)
                    else:
                        # multiple candidate might match with previous object, here we choose the one with
                        # largest comprehensive score
                        obj_id = match_id - 1
                        match_score = comp_scores[idx, match_id]
                        if match_score > best_match_scores[obj_id]:
                            det_obj_ids[idx] = obj_id
                            best_match_scores[obj_id] = match_score
                            # udpate feature
                            self.prev_roi_feats[obj_id] = det_roi_feats[idx]
                            self.prev_bboxes[obj_id] = det_bboxes[0][idx]

        obj_segms = {}
        masks = det_bboxes[2]
        for i in range(det_bboxes[0].shape[0]):
            label = det_labels[i]
            mask = masks[i].cpu().numpy()
            im_mask = np.zeros((ori_shape[0], ori_shape[1]), dtype=np.uint8)
            shape = np.minimum(mask.shape, ori_shape[0:2])
            im_mask[:shape[0], :shape[1]] = mask[:shape[0], :shape[1]]
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            if det_obj_ids[i] >= 0:
                obj_segms[det_obj_ids[i]] = rle

        result_list.append([det_bboxes[0], det_bboxes[1], obj_segms, det_obj_ids])

        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          centernesses,
                          cof_preds,
                          feat_mask,
                          mlvl_points,
                          img_shape,
                          ori_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        mlvl_cofs = []
        for cls_score, bbox_pred, cof_pred, centerness, points in zip(
                cls_scores, bbox_preds, cof_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            cof_pred = cof_pred.permute(1, 2, 0).reshape(-1, 32 * 4)

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                cof_pred = cof_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_cofs.append(cof_pred)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_cofs = torch.cat(mlvl_cofs)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        mlvl_scores = mlvl_scores * mlvl_centerness.view(-1, 1)
        det_bboxes, det_labels, det_cofs = self.fast_nms(mlvl_bboxes, mlvl_scores[:, 1:].transpose(1, 0).contiguous(),
                                                         mlvl_cofs, cfg, iou_threshold=0.5)
        masks = []
        if det_bboxes.shape[0] > 0:
            scale = 2
            #####spp########################
            img_mask1 = feat_mask.permute(1, 2, 0)
            pos_masks00 = torch.sigmoid(img_mask1 @ det_cofs[:, 0:32].t())
            pos_masks01 = torch.sigmoid(img_mask1 @ det_cofs[:, 32:64].t())
            pos_masks10 = torch.sigmoid(img_mask1 @ det_cofs[:, 64:96].t())
            pos_masks11 = torch.sigmoid(img_mask1 @ det_cofs[:, 96:128].t())
            if rescale:
                pos_masks = torch.stack([pos_masks00, pos_masks01, pos_masks10, pos_masks11], dim=0)
                pos_masks = self.crop_cuda(pos_masks, det_bboxes[:, :4] * det_bboxes.new_tensor(scale_factor) / scale)
                # pos_masks = crop_split(pos_masks00, pos_masks01, pos_masks10, pos_masks11, det_bboxes * det_bboxes.new_tensor(scale_factor) / scale)
            else:
                pos_masks = torch.stack([pos_masks00, pos_masks01, pos_masks10, pos_masks11], dim=0)
                pos_masks = self.crop_cuda(pos_masks, det_bboxes[:, :4] / scale)
                # pos_masks = crop_split(pos_masks00, pos_masks01, pos_masks10, pos_masks11, det_bboxes / scale)
            pos_masks = pos_masks.permute(2, 0, 1)
            if rescale:
                masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale / scale_factor, mode='bilinear',
                                      align_corners=False).squeeze(0)
            else:
                masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale, mode='bilinear',
                                      align_corners=False).squeeze(0)
            masks.gt_(0.5)

        return det_bboxes, det_labels, masks

    def extract_box_feature_center_single(self, track_feats, gt_bboxs):

        track_box_feats = track_feats.new_zeros(gt_bboxs.size()[0], 512)

        #####extract feature box############
        ref_feat_stride = 8
        gt_center_xs = torch.floor((gt_bboxs[:, 2] + gt_bboxs[:, 0]) / 2.0 / ref_feat_stride).long()
        gt_center_ys = torch.floor((gt_bboxs[:, 3] + gt_bboxs[:, 1]) / 2.0 / ref_feat_stride).long()

        aa = track_feats.permute(1, 2, 0)
        bb = aa[gt_center_ys, gt_center_xs, :]
        track_box_feats += bb

        return track_box_feats

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            dtype (torch.dtype): Type of points.
            device (torch.device): Device of points.

        Returns:
            tuple: points of each image.
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            points = self.get_points_single(featmap_sizes[i], self.strides[i],
                                            dtype, device)
            mlvl_points.append(points)

        return mlvl_points

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2

        return points

    def fcos_target(self, points, gt_bboxes_list, gt_labels_list):
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list, gt_inds = multi_apply(
            self.fcos_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            concat_lvl_bbox_targets.append(
                torch.cat(
                    [bbox_targets[i] for bbox_targets in bbox_targets_list]))
        return concat_lvl_labels, concat_lvl_bbox_targets, labels_list, bbox_targets_list, gt_inds

    def fcos_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        if num_gts == 0:
            return gt_labels.new_zeros(num_points), \
                   gt_bboxes.new_zeros((num_points, 4))

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0] + 1) * (
                gt_bboxes[:, 3] - gt_bboxes[:, 1] + 1)
        # TODO: figure out why these two are different
        # areas = areas[None].expand(num_points, num_gts)
        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        bbox_targets = bbox_targets

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # condition2: limit the regression range for each location
        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
                                       max_regress_distance >= regress_ranges[..., 0]) & (
                                       max_regress_distance <= regress_ranges[..., 1])

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = 0
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        gt_ind = min_area_inds[labels > 0]

        return labels, bbox_targets, gt_ind

    def centerness_target(self, pos_bbox_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
                                     left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                                     top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def fast_nms(self, boxes, scores, masks, cfg, iou_threshold=0.5, top_k=200):
        scores, idx = scores.sort(1, descending=True)

        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]

        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = self.jaccard(boxes, boxes)
        iou.triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= iou_threshold)

        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        keep *= (scores > cfg.score_thr)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_per_img]
        scores = scores[:cfg.max_per_img]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
        boxes = torch.cat([boxes, scores[:, None]], dim=1)
        return boxes, classes, masks

    def jaccard(self, box_a, box_b, iscrowd: bool = False):
        """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
        is simply the intersection over union of two boxes.  Here we operate on
        ground truth boxes and default boxes. If iscrowd=True, put the crowd in box_b.
        E.g.:
            A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
        Args:
            box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
            box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
        Return:
            jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
        """
        use_batch = True
        if box_a.dim() == 2:
            use_batch = False
            box_a = box_a[None, ...]
            box_b = box_b[None, ...]

        inter = self.intersect(box_a, box_b)
        area_a = ((box_a[:, :, 2] - box_a[:, :, 0]) *
                  (box_a[:, :, 3] - box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, :, 2] - box_b[:, :, 0]) *
                  (box_b[:, :, 3] - box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter

        out = inter / area_a if iscrowd else inter / union
        return out if use_batch else out.squeeze(0)

    def intersect(self, box_a, box_b):
        """ We resize both tensors to [A,B,2] without new malloc:
        [A,2] -> [A,1,2] -> [A,B,2]
        [B,2] -> [1,B,2] -> [A,B,2]
        Then we compute the area of intersect between box_a and box_b.
        Args:
          box_a: (tensor) bounding boxes, Shape: [n,A,4].
          box_b: (tensor) bounding boxes, Shape: [n,B,4].
        Return:
          (tensor) intersection area, Shape: [n,A,B].
        """
        n = box_a.size(0)
        A = box_a.size(1)
        B = box_b.size(1)
        max_xy = torch.min(box_a[:, :, 2:].unsqueeze(2).expand(n, A, B, 2),
                           box_b[:, :, 2:].unsqueeze(1).expand(n, A, B, 2))
        min_xy = torch.max(box_a[:, :, :2].unsqueeze(2).expand(n, A, B, 2),
                           box_b[:, :, :2].unsqueeze(1).expand(n, A, B, 2))
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, :, 0] * inter[:, :, :, 1]