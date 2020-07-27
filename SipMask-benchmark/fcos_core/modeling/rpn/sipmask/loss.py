"""
This file contains specific functions for computing losses of SipMask
file
"""

import torch
from torch.nn import functional as F
from torch import nn
import os
from ..utils import concat_box_prediction_layers
from fcos_core.layers import IOULoss
from fcos_core.layers import SigmoidFocalLoss
from fcos_core.modeling.matcher import Matcher
from fcos_core.modeling.utils import cat
from fcos_core.structures.boxlist_ops import boxlist_iou
from fcos_core.structures.boxlist_ops import cat_boxlist
import numpy as np
from fcos_core.layers import CropSplit
from fcos_core.layers import CropSplitGt
from fcos_core.layers import nms as _box_nms
INF = 100000000


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious

def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    import torch.distributed as dist
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h



class SipMaskLossComputation(object):
    """
    This class computes the SipMasK losses.
    """

    def __init__(self, cfg):
        self.cls_loss_func = SigmoidFocalLoss(
            cfg.MODEL.SIPMASK.LOSS_GAMMA,
            cfg.MODEL.SIPMASK.LOSS_ALPHA
        )
        self.fpn_strides = cfg.MODEL.SIPMASK.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.SIPMASK.CENTER_SAMPLING_RADIUS
        self.iou_loss_type = cfg.MODEL.SIPMASK.IOU_LOSS_TYPE
        self.norm_reg_targets = cfg.MODEL.SIPMASK.NORM_REG_TARGETS

        # we make use of IOU Loss for bounding boxes regression,
        # but we found that L1 in log scale can yield a similar performance
        self.box_reg_loss_func = IOULoss(self.iou_loss_type)
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.crop_cuda = CropSplit(2)
        self.crop_gt_cuda = CropSplitGt(2)

    def get_sample_region(self, gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0]
        K = len(gt_xs)
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        for level, n_p in enumerate(num_points_per):
            end = beg + n_p
            stride = strides[level] * radius
            xmin = center_x[beg:end] - stride
            ymin = center_y[beg:end] - stride
            xmax = center_x[beg:end] + stride
            ymax = center_y[beg:end] + stride
            # limit sample region in gt
            center_gt[beg:end, :, 0] = torch.where(
                xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
            )
            center_gt[beg:end, :, 1] = torch.where(
                ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
            )
            center_gt[beg:end, :, 2] = torch.where(
                xmax > gt[beg:end, :, 2],
                gt[beg:end, :, 2], xmax
            )
            center_gt[beg:end, :, 3] = torch.where(
                ymax > gt[beg:end, :, 3],
                gt[beg:end, :, 3], ymax
            )
            beg = end
        left = gt_xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def prepare_targets(self, points, targets):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]

        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )

        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets, gt_targets, gt_inds = self.compute_targets_for_locations(
            points_all_level, targets, expanded_object_sizes_of_interest
        )

        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first, labels, gt_targets, gt_inds

    def compute_targets_for_locations(self, locations, targets, object_sizes_of_interest):
        labels = []
        reg_targets = []
        gt_targets = []
        gt_inds = []
        xs, ys = locations[:, 0], locations[:, 1]

        for im_i in range(len(targets)):
            targets_per_im = targets[im_i]
            assert targets_per_im.mode == "xyxy"
            bboxes = targets_per_im.bbox
            labels_per_im = targets_per_im.get_field("labels")
            area = targets_per_im.area()

            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            r = bboxes[:, 2][None] - xs[:, None]
            b = bboxes[:, 3][None] - ys[:, None]
            reg_targets_per_im = torch.stack([l, t, r, b], dim=2)
            

            if self.center_sampling_radius > 0:
                is_in_boxes = self.get_sample_region(
                    bboxes,
                    self.fpn_strides,
                    self.num_points_per_level,
                    xs, ys,
                    radius=self.center_sampling_radius
                )
            else:
                # no center sampling, it will use all the locations within a ground-truth box
                is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

            max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
            # limit the regression range for each location
            is_cared_in_the_level = \
                (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
                (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_the_level == 0] = INF

            # if there are still more than one objects for a location,
            # we choose the one with minimal area
            locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

            reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
            labels_per_im = labels_per_im[locations_to_gt_inds]
            labels_per_im[locations_to_min_area == INF] = 0

            labels.append(labels_per_im)
            reg_targets.append(reg_targets_per_im)

            gt_target = bboxes[locations_to_gt_inds]
            gt_targets.append(gt_target)
            gt_inds.append(locations_to_gt_inds[labels_per_im>0])

        return labels, reg_targets, gt_targets, gt_inds

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)


    def decode_for_single_feature_map(self, locations, box_regression, fpn_stride):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, A, H, W = box_regression.shape

        # put in the same format as locations
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)*fpn_stride
        results = []
        for i in range(N):
            per_box_regression = box_regression[i]
            per_locations = locations

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)
            results.append(detections)
        # print(len(detections))
        return results

    def __call__(self, locations, box_cls, box_regression, centerness, cof_preds, feat_mask, targets):
        """
        Arguments:
            locations (list[BoxList])
            box_cls (list[Tensor])
            box_regression (list[Tensor])
            centerness (list[Tensor])
            targets (list[BoxList])

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        N = box_cls[0].size(0)
        num_classes = box_cls[0].size(1)
        labels, reg_targets, labels_list, bbox_gt_list, gt_inds = self.prepare_targets(locations, targets)

        ######decode box########
        sampled_boxes = []
        for _, (l, b, s) in enumerate(zip(locations, box_regression, self.fpn_strides)):
            sampled_boxes.append(self.decode_for_single_feature_map(l, b, s))

        flatten_sampled_boxes = [
            torch.cat([labels_level_img.reshape(-1, 4)
                       for labels_level_img in sampled_boxes_per_img])
            for sampled_boxes_per_img in zip(*sampled_boxes)
        ]


        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        cls_loss = self.cls_loss_func(
            box_cls_flatten,
            labels_flatten.int()
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            ) / sum_centerness_targets_avg_per_gpu
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()


        ##########mask loss#################
        num_imgs = len(flatten_sampled_boxes)
        flatten_cls_scores1 = []
        for l in range(len(labels)):
            flatten_cls_scores1.append(box_cls[l].permute(0, 2, 3, 1).reshape(num_imgs, -1, num_classes))

        flatten_cls_scores1 = torch.cat(flatten_cls_scores1,dim=1)

        flatten_cof_preds = [
            cof_pred.permute(0, 2, 3, 1).reshape(len(labels_list),-1, 32*4)
            for cof_pred in cof_preds
        ]
        flatten_cof_preds = torch.cat(flatten_cof_preds,dim=1)

        loss_mask = 0


        for i in range(num_imgs):
            labels = torch.cat([labels_level.flatten() for labels_level in labels_list[i]])
            # bbox_gt = torch.cat([gt_level.reshape(-1,4) for gt_level in bbox_gt_list[i]])
            bbox_dt = flatten_sampled_boxes[i]/2
            bbox_dt = bbox_dt.detach()
            pos_inds = labels > 0

            # # print(pos_inds.shape)
            cof_pred = flatten_cof_preds[i][pos_inds]
            img_mask = feat_mask[i]
            mask_h = feat_mask[i].shape[1]
            mask_w = feat_mask[i].shape[2]
            idx_gt = gt_inds[i]
            bbox_dt = bbox_dt[pos_inds, :4]
            gt_masks = targets[i].get_field("masks").get_mask_tensor().to(dtype=torch.float32,device=feat_mask.device)
            gt_masks = gt_masks.reshape(-1,gt_masks.shape[-2],gt_masks.shape[-1])

            # print(torch.max(bbox_dt[:,2]),torch.max(bbox_dt[:,3]))
            area = (bbox_dt[:,2]-bbox_dt[:,0])*(bbox_dt[:,3]-bbox_dt[:,1])
            bbox_dt = bbox_dt[area>0,:]
            idx_gt = idx_gt[area>0]
            cof_pred = cof_pred[area>0]
            if bbox_dt.shape[0]==0:
                continue


            bbox_gt = targets[i].bbox
            cls_score = flatten_cls_scores1[i, pos_inds, labels[pos_inds] - 1].sigmoid().detach()
            cls_score = cls_score[area>0]
            ious = bbox_overlaps(bbox_gt[idx_gt]/2, bbox_dt, is_aligned=True)
            weighting = cls_score * ious
            weighting = weighting/torch.sum(weighting)*len(weighting)
            keep = _box_nms(bbox_dt, cls_score, 0.9)
            bbox_dt = bbox_dt[keep]
            weighting = weighting[keep]
            idx_gt = idx_gt[keep]
            cof_pred = cof_pred[keep]

            gt_mask = F.interpolate(gt_masks.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)

            shape = np.minimum(feat_mask[i].shape, gt_mask.shape)
            gt_mask_new = gt_mask.new_zeros(gt_mask.shape[0], mask_h, mask_w)
            gt_mask_new[:gt_mask.shape[0], :shape[1], :shape[2]] = gt_mask[:gt_mask.shape[0], :shape[1], :shape[2]]
            gt_mask_new = gt_mask_new.gt(0.5).float()

            gt_mask_new = torch.index_select(gt_mask_new,0,idx_gt).permute(1, 2, 0).contiguous()

            #######spp###########################
            img_mask1 = img_mask.permute(1,2,0)
            pos_masks00 = torch.sigmoid(img_mask1 @ cof_pred[:, 0:32].t())
            pos_masks01 = torch.sigmoid(img_mask1 @ cof_pred[:, 32:64].t())
            pos_masks10 = torch.sigmoid(img_mask1 @ cof_pred[:, 64:96].t())
            pos_masks11 = torch.sigmoid(img_mask1 @ cof_pred[:, 96:128].t())
            pred_masks = torch.stack([pos_masks00, pos_masks01, pos_masks10, pos_masks11], dim=0)
            pred_masks = self.crop_cuda(pred_masks, bbox_dt)
            gt_mask_crop = self.crop_gt_cuda(gt_mask_new, bbox_dt)

            pre_loss = F.binary_cross_entropy(pred_masks, gt_mask_crop, reduction='none')

            pos_get_csize = center_size(bbox_dt)
            gt_box_width = pos_get_csize[:, 2]
            gt_box_height = pos_get_csize[:, 3]
            pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height / pos_get_csize.shape[0]
            loss_mask += torch.sum(pre_loss*weighting.detach())


        loss_mask = loss_mask/num_imgs
        if loss_mask > 1.0:
            loss_mask = loss_mask*0.5

        return cls_loss, reg_loss, centerness_loss, loss_mask


def make_sipmask_loss_evaluator(cfg):
    loss_evaluator = SipMaskLossComputation(cfg)
    return loss_evaluator
