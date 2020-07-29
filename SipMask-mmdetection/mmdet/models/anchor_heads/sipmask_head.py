import torch, mmcv
import torch.nn as nn
from mmcv.cnn import normal_init, kaiming_init

from mmdet.core import distance2bbox, bbox_overlaps, force_fp32, multi_apply, multiclass_nms, multiclass_nms_idx
from mmdet.ops import ConvModule, Scale
from ..builder import build_loss
from ..registry import HEADS
from ..utils import bias_init_with_prob
from mmdet.ops import DeformConv, CropSplit, CropSplitGt
import torch.nn.functional as F
import pycocotools.mask as mask_util
import numpy as np
INF = 1e8

def center_size(boxes):
    return torch.cat(( (boxes[:, 2:] + boxes[:, :2])/2,     # cx, cy
                        boxes[:, 2:] - boxes[:, :2]  ), 1)  # w, h


class FeatureAlign(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=4,
                 flag_norm=True):
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
        self.flag_norm = flag_norm



    def init_weights(self,bias_value=0):
        torch.nn.init.normal_(self.conv_offset.weight, std=0.0)
        torch.nn.init.normal_(self.conv_adaption.weight, std=0.01)

    def forward(self, x, shape):
        offset = self.conv_offset(shape.detach())
        if self.flag_norm:
            x = self.relu(self.norm(self.conv_adaption(x, offset)))
        else:
            x = self.relu(self.conv_adaption(x, offset))
        return x


def crop_split(masks00, masks01, masks10, masks11, boxes, masksG=None):
    h, w, n = masks00.size()
    rows = torch.arange(w, device=masks00.device, dtype=boxes.dtype).view(1, -1, 1).expand(h, w, n)
    cols = torch.arange(h, device=masks00.device, dtype=boxes.dtype).view(-1, 1, 1).expand(h, w, n)


    x1, x2 = boxes[:, 0], boxes[:, 2]
    y1, y2 = boxes[:, 1], boxes[:, 3]
    xc = (x1+x2)/2
    yc = (y1+y2)/2
    x1 = torch.clamp(x1, min=0, max=w - 1)
    y1 = torch.clamp(y1, min=0, max=h - 1)
    x2 = torch.clamp(x2, min=0, max=w - 1)
    y2 = torch.clamp(y2, min=0, max=h - 1)
    xc = torch.clamp(xc, min=0, max=w - 1)
    yc = torch.clamp(yc, min=0, max=h - 1)

    ##x1,y1,xc,yc
    crop_mask = (rows >= x1.view(1, 1, -1)) & (rows < xc.view(1, 1, -1)) & (cols >= y1.view(1, 1, -1)) & (cols < yc.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks00 = masks00 * crop_mask

    ##xc,y1,x2,yc
    crop_mask = (rows >= xc.view(1, 1, -1)) & (rows < x2.view(1, 1, -1)) & (cols >= y1.view(1, 1, -1)) & (cols < yc.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks01 = masks01 * crop_mask

    ##x1,yc,xc,y2
    crop_mask = (rows >= x1.view(1, 1, -1)) & (rows < xc.view(1, 1, -1)) & (cols >= yc.view(1, 1, -1)) & (cols < y2.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks10 = masks10 * crop_mask

    ##xc,yc,x2,y2
    crop_mask = (rows >= xc.view(1, 1, -1)) & (rows < x2.view(1, 1, -1)) & (cols >= yc.view(1, 1, -1)) & (cols < y2.view(1, 1, -1))
    crop_mask = crop_mask.float().detach()
    masks11 = masks11 * crop_mask

    masks = masks00+masks01+masks10+masks11

    ########whole
    if masksG is not None:
        crop_mask = (rows >= x1.view(1, 1, -1)) & (rows < x2.view(1, 1, -1)) & (cols >= y1.view(1, 1, -1)) & (cols < y2.view(1, 1, -1))
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
                 ssd_flag=False,
                 rescoring_flag=False,
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
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)):#
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
        self.loss_center = build_loss(loss_bbox)
        self.ssd_flag = ssd_flag
        self.rescoring_flag = rescoring_flag
        if self.rescoring_flag:
            self.loss_iou = build_loss(dict(type='MSELoss', loss_weight=1.0, reduction='sum'))
        self._init_layers()

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs-1):
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
        self.feat_align = FeatureAlign(self.feat_channels, self.feat_channels, 3, flag_norm=self.norm_cfg is not None)
        self.sip_cof = nn.Conv2d(self.feat_channels, self.nc*4, 3, padding=1)

        self.sip_mask_lat = nn.Conv2d(512, self.nc, 3, padding=1)
        self.sip_mask_lat0 = nn.Conv2d(768, 512, 1, padding=0)

        if self.rescoring_flag:
            self.convs_scoring = []
            channels = [1, 16, 16, 16, 32, 64, 128]
            for i in range(6):
                in_channels = channels[i]
                out_channels = channels[i + 1]
                stride = 2 if i == 0 else 2
                padding = 0
                self.convs_scoring.append(
                    ConvModule(
                        in_channels, out_channels,
                        3,
                        stride=stride,
                        padding=padding,
                        bias=True))
            self.convs_scoring = nn.Sequential(*self.convs_scoring)
            self.mask_scoring = nn.Conv2d(128, self.num_classes-1, 1)
            for m in self.convs_scoring:
                kaiming_init(m.conv)
            normal_init(self.mask_scoring, std=0.001)

        self.relu = nn.ReLU(inplace=True)
        self.crop_cuda = CropSplit(2)
        self.crop_gt_cuda = CropSplitGt(2)
        self.init_weights()

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.fcos_cls, std=0.01, bias=bias_cls)
        normal_init(self.fcos_reg, std=0.01)
        normal_init(self.fcos_centerness, std=0.01)

        normal_init(self.sip_cof, std=0.001)
        normal_init(self.sip_mask_lat, std=0.01)
        normal_init(self.sip_mask_lat0, std=0.01)
        self.feat_align.init_weights()

    def forward(self, feats):
        # return multi_apply(self.forward_single, feats, self.scales)
        cls_scores = []
        bbox_preds = []
        centernesses = []
        cof_preds = []
        feat_masks = []
        count = 0

        for x, scale, stride in zip(feats,self.scales, self.strides):
            cls_feat = x
            reg_feat = x
            for cls_layer in self.cls_convs:
                cls_feat = cls_layer(cls_feat)

            for reg_layer in self.reg_convs:
                reg_feat = reg_layer(reg_feat)

            # scale the bbox_pred of different level
            # float to avoid overflow when enabling FP16
            bbox_pred = scale(self.fcos_reg(reg_feat))

            cls_feat = self.feat_align(cls_feat, bbox_pred)
            cls_score = self.fcos_cls(cls_feat)
            centerness = self.fcos_centerness(reg_feat)
            centernesses.append(centerness)
            cls_scores.append(cls_score)
            bbox_preds.append(bbox_pred.float()*stride)

            ########COFFECIENTS###############
            cof_pred = self.sip_cof(cls_feat)
            cof_preds.append(cof_pred)

            ############contextual#######################
            if count < 3:
                if count == 0:
                    feat_masks.append(reg_feat)
                else:
                    feat_up = F.interpolate(reg_feat, scale_factor=(2 ** count), mode='bilinear', align_corners=False)
                    feat_masks.append(feat_up)
            count = count + 1
        # ################contextual enhanced##################
        feat_masks = torch.cat(feat_masks, dim=1)
        feat_masks = self.relu(self.sip_mask_lat(self.relu(self.sip_mask_lat0(feat_masks))))
        feat_masks = F.interpolate(feat_masks, scale_factor=4, mode='bilinear', align_corners=False)

        return cls_scores, bbox_preds, centernesses, cof_preds, feat_masks

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             cof_preds,
             feat_masks,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None,
             gt_masks_list=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        all_level_points, all_level_strides = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        labels, bbox_targets, label_list, bbox_targets_list, gt_inds = self.fcos_target(all_level_points,
                                                                                        gt_bboxes, gt_labels)
        #decode detection and groundtruth
        det_bboxes = []
        det_targets = []
        num_levels = len(bbox_preds)

        for img_id in range(len(img_metas)):
            bbox_pred_list = [
                bbox_preds[i][img_id].permute(1, 2, 0).reshape(-1, 4).detach() for i in range(num_levels)
            ]
            bbox_target_list =  bbox_targets_list[img_id]

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
            gt_masks.append(torch.from_numpy(np.array(gt_masks_list[i][:gt_label.shape[0]], dtype=np.float32)).to(gt_label.device))

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
        flatten_strides = torch.cat(
            [strides.view(-1,1).repeat(num_imgs, 1) for strides in all_level_strides])

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
            pos_strides = flatten_strides[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds/pos_strides)
            pos_decoded_target_preds = distance2bbox(pos_points,
                                                     pos_bbox_targets/pos_strides)
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
            cls_score.permute(0, 2, 3, 1).reshape(num_imgs,-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores1 = torch.cat(flatten_cls_scores1,dim=1)

        flatten_cof_preds = [
            cof_pred.permute(0, 2, 3, 1).reshape(cof_pred.shape[0],-1, 32*4)
            for cof_pred in cof_preds
        ]

        loss_mask = 0
        loss_iou = 0
        num_iou = 0.1
        flatten_cof_preds = torch.cat(flatten_cof_preds,dim=1)
        for i in range(num_imgs):
            labels = torch.cat([labels_level.flatten() for labels_level in label_list[i]])
            bbox_dt = det_bboxes[i]/2
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
            if bbox_dt.shape[0] == 0:
                loss_mask += area.sum()*0
                continue

            bbox_gt = gt_bboxes[i]
            cls_score = flatten_cls_scores1[i, pos_inds, labels[pos_inds] - 1].sigmoid().detach()
            cls_score = cls_score[area>1.0]
            pos_inds = pos_inds[area > 1.0]
            ious = bbox_overlaps(bbox_gt[idx_gt]/2, bbox_dt, is_aligned=True)
            with torch.no_grad():
                weighting = cls_score * ious
                weighting = weighting/(torch.sum(weighting)+0.0001)*len(weighting)

            gt_mask = F.interpolate(gt_masks[i].unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False).squeeze(0)

            shape = np.minimum(feat_masks[i].shape, gt_mask.shape)
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
            # pred_masks, gt_mask_crop = crop_split(pos_masks00, pos_masks01, pos_masks10, pos_masks11, bbox_dt,
            #                                       gt_mask_new)

            pre_loss = F.binary_cross_entropy(pred_masks, gt_mask_crop, reduction='none')
            pos_get_csize = center_size(bbox_dt)
            gt_box_width = pos_get_csize[:, 2]
            gt_box_height = pos_get_csize[:, 3]
            pre_loss = pre_loss.sum(dim=(0, 1)) / gt_box_width / gt_box_height / pos_get_csize.shape[0]
            loss_mask += torch.sum(pre_loss*weighting.detach())


            if self.rescoring_flag:
                pos_labels = labels[pos_inds] - 1
                input_iou = pred_masks.detach().unsqueeze(0).permute(3, 0, 1, 2)
                pred_iou = self.convs_scoring(input_iou)
                pred_iou = self.relu(self.mask_scoring(pred_iou))
                pred_iou = F.max_pool2d(pred_iou, kernel_size=pred_iou.size()[2:]).squeeze(-1).squeeze(-1)
                pred_iou = pred_iou[range(pred_iou.size(0)), pos_labels]
                with torch.no_grad():
                    mask_pred = (pred_masks > 0.4).float()
                    mask_pred_areas = mask_pred.sum((0, 1))
                    overlap_areas = (mask_pred * gt_mask_new).sum((0, 1))
                    gt_full_areas = gt_mask_new.sum((0, 1))
                    iou_targets = overlap_areas / (mask_pred_areas + gt_full_areas - overlap_areas + 0.1)

                    iou_weights = ((iou_targets > 0.1) & (iou_targets <= 1.0) & (gt_full_areas >= 10 * 10)).float()

                loss_iou += self.loss_iou(pred_iou.view(-1, 1), iou_targets.view(-1, 1), iou_weights.view(-1, 1))
                num_iou += torch.sum(iou_weights.detach())
        loss_mask = loss_mask/num_imgs
        if self.rescoring_flag:
            loss_iou = loss_iou * 10 / num_iou.detach()
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness,
                loss_mask=loss_mask,
                loss_iou=loss_iou)
        else:
            return dict(
                loss_cls=loss_cls,
                loss_bbox=loss_bbox,
                loss_centerness=loss_centerness,
                loss_mask=loss_mask)

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   cof_preds,
                   feat_masks,
                   img_metas,
                   cfg,
                   rescale=None):
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points, mlvl_strides = self.get_points(featmap_sizes, bbox_preds[0].dtype,
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

            img_shape = img_metas[img_id]['img_shape']
            ori_shape = img_metas[img_id]['ori_shape']
            scale_factor = img_metas[img_id]['scale_factor']

            det_bboxes = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                                centerness_pred_list, cof_pred_list, feat_mask_list,
                                                mlvl_points, img_shape, ori_shape,
                                                scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
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
            cof_pred = cof_pred.permute(1,2,0).reshape(-1,32*4)

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

        if self.ssd_flag is False:
            det_bboxes, det_labels, idxs_keep = multiclass_nms_idx(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness)
        else:
            mlvl_scores = mlvl_scores*mlvl_centerness.view(-1,1)
            det_bboxes, det_labels, det_cofs = self.fast_nms(mlvl_bboxes, mlvl_scores[:, 1:].transpose(1, 0).contiguous(),
                                                             mlvl_cofs, iou_threshold=cfg.nms.iou_thr, score_thr=cfg.score_thr)

        cls_segms = [[] for _ in range(self.num_classes - 1)]
        mask_scores = [[] for _ in range(self.num_classes - 1)]
        if det_bboxes.shape[0]>0:
            scale = 2

            if self.ssd_flag is False:
                det_cofs = mlvl_cofs[idxs_keep]
            #####spp########################
            img_mask1 = feat_mask.permute(1,2,0)
            pos_masks00 = torch.sigmoid(img_mask1 @ det_cofs[:, 0:32].t())
            pos_masks01 = torch.sigmoid(img_mask1 @ det_cofs[:, 32:64].t())
            pos_masks10 = torch.sigmoid(img_mask1 @ det_cofs[:, 64:96].t())
            pos_masks11 = torch.sigmoid(img_mask1 @ det_cofs[:, 96:128].t())
            pos_masks = torch.stack([pos_masks00,pos_masks01,pos_masks10,pos_masks11],dim=0)
            pos_masks = self.crop_cuda(pos_masks, det_bboxes[:,:4] * det_bboxes.new_tensor(scale_factor) / scale)
            # pos_masks = crop_split(pos_masks00, pos_masks01, pos_masks10, pos_masks11,
            #                        det_bboxes * det_bboxes.new_tensor(scale_factor) / scale)

            pos_masks = pos_masks.permute(2, 0, 1)
            # masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale/scale_factor, mode='bilinear', align_corners=False).squeeze(0)
            if self.ssd_flag:
                masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale / scale_factor[3:1:-1], mode='bilinear', align_corners=False).squeeze(0)
            else:
                masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale / scale_factor, mode='bilinear', align_corners=False).squeeze(0)
            masks.gt_(0.4)

            if self.rescoring_flag:
                pred_iou = pos_masks.unsqueeze(1)
                pred_iou = self.convs_scoring(pred_iou)
                pred_iou = self.relu(self.mask_scoring(pred_iou))
                pred_iou = F.max_pool2d(pred_iou, kernel_size=pred_iou.size()[2:]).squeeze(-1).squeeze(-1)
                pred_iou = pred_iou[range(pred_iou.size(0)), det_labels].squeeze()
                mask_scores = pred_iou*det_bboxes[:, -1]
                mask_scores = mask_scores.cpu().numpy()
                mask_scores = [mask_scores[det_labels.cpu().numpy() == i] for i in range(self.num_classes - 1)]

        for i in range(det_bboxes.shape[0]):
            label = det_labels[i]
            mask = masks[i].cpu().numpy()
            im_mask = np.zeros((ori_shape[0], ori_shape[1]), dtype=np.uint8)
            shape = np.minimum(mask.shape, ori_shape[0:2])
            im_mask[:shape[0],:shape[1]] = mask[:shape[0],:shape[1]]
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            cls_segms[label].append(rle)

        if self.rescoring_flag:
            return det_bboxes, det_labels, (cls_segms, mask_scores)
        else:
            return det_bboxes, det_labels, cls_segms

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
        mlvl_strides = []
        for i in range(len(featmap_sizes)):
            points, strides = self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device)
            mlvl_points.append(points)
            mlvl_strides.append(strides)

        return mlvl_points, mlvl_strides

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
        strides = points[:,0]*0+stride
        return points, strides

    def center_target(self, gt_bboxes_raw, gt_masks_raw, featmap_size):
        stride = 8
        h, w = featmap_size
        x_range = torch.arange(0, w, 1, dtype=gt_bboxes_raw[0].dtype, device=gt_bboxes_raw[0].device)
        y_range = torch.arange(0, h, 1, dtype=gt_bboxes_raw[0].dtype, device=gt_bboxes_raw[0].device)
        y, x = torch.meshgrid(y_range, x_range)
        center_targets = []
        labels = []
        for n in range(len(gt_bboxes_raw)):
            center_target = gt_bboxes_raw[n].new(featmap_size[0], featmap_size[1],4) + 0
            label = gt_bboxes_raw[n].new_zeros(featmap_size)
            gt_bboxes = gt_bboxes_raw[n]/stride
            gt_masks = gt_masks_raw[n]
            mask_size = gt_masks.shape
            pos_left = torch.floor(gt_bboxes[:, 0]).long().clamp(0, gt_masks.shape[2]//stride - 1)
            pos_right = torch.ceil(gt_bboxes[:, 2]).long().clamp(0, gt_masks.shape[2]//stride - 1)
            pos_top = torch.floor(gt_bboxes[:, 1]).long().clamp(0, gt_masks.shape[1]//stride - 1)
            pos_down = torch.ceil(gt_bboxes[:, 3]).long().clamp(0, gt_masks.shape[1]//stride - 1)
            for px1, py1, px2, py2, gt_mask, (x1, y1, x2, y2) in \
                    zip(pos_left, pos_top, pos_right, pos_down, gt_masks, gt_bboxes):
                gt_mask = mmcv.imrescale(gt_mask, scale=1. / stride)
                gt_mask = torch.Tensor(gt_mask)

                label[py1:py2 + 1, px1:px2 + 1] = gt_mask[py1:py2 + 1, px1:px2 + 1]
                center_target[py1:py2 + 1, px1:px2 + 1, 0] = x1 / w
                center_target[py1:py2 + 1, px1:px2 + 1, 1] = y1 / h
                center_target[py1:py2 + 1, px1:px2 + 1, 2] = x2 / w
                center_target[py1:py2 + 1, px1:px2 + 1, 3] = y2 / h
            center_targets.append(center_target.reshape(-1, 4))
            labels.append(label.reshape(-1, 1))
        labels = torch.cat(labels)
        center_targets = torch.cat(center_targets)
        return labels, center_targets

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

    def fast_nms(self, boxes, scores, masks, iou_threshold = 0.5, top_k = 200, score_thr=0.1):
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
        keep *= (scores > score_thr)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:100]
        scores = scores[:100]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
        boxes = torch.cat([boxes, scores[:, None]], dim=1)
        return boxes, classes, masks

    def jaccard(self, box_a, box_b, iscrowd:bool=False):
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
        area_a = ((box_a[:, :, 2]-box_a[:, :, 0]) *
                  (box_a[:, :, 3]-box_a[:, :, 1])).unsqueeze(2).expand_as(inter)  # [A,B]
        area_b = ((box_b[:, :, 2]-box_b[:, :, 0]) *
                  (box_b[:, :, 3]-box_b[:, :, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
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