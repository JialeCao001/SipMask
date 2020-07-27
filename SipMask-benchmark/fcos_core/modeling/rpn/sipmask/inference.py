import torch

from ..inference import RPNPostProcessor
from ..utils import permute_and_flatten
import torch.nn.functional as F
import numpy as np
import pycocotools.mask as mask_util
from fcos_core.modeling.box_coder import BoxCoder
from fcos_core.modeling.utils import cat
from fcos_core.structures.bounding_box import BoxList
from fcos_core.structures.boxlist_ops import cat_boxlist
from fcos_core.structures.boxlist_ops import boxlist_ml_nms
from fcos_core.structures.boxlist_ops import remove_small_boxes
from fcos_core.layers import CropSplit


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

class SipMaskPostProcessor(torch.nn.Module):
    """
    Performs post-processing on the outputs of the RetinaNet boxes.
    This is only used in the testing.
    """
    def __init__(
        self,
        pre_nms_thresh,
        pre_nms_top_n,
        nms_thresh,
        fpn_post_nms_top_n,
        min_size,
        num_classes,
        bbox_aug_enabled=False
    ):
        """
        Arguments:
            pre_nms_thresh (float)
            pre_nms_top_n (int)
            nms_thresh (float)
            fpn_post_nms_top_n (int)
            min_size (int)
            num_classes (int)
            box_coder (BoxCoder)
        """
        super(SipMaskPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        self.bbox_aug_enabled = bbox_aug_enabled
        self.crop_cuda = CropSplit(2)
        self.crop_cuda0 = CropSplit(1)
        # self.crop_cuda = CropSplit(2)


    def forward_for_single_feature_map(
            self, locations, box_cls,
            box_regression, centerness,
            det_cofs, image_sizes):
        """
        Arguments:
            anchors: list[BoxList]
            box_cls: tensor of size N, A * C, H, W
            box_regression: tensor of size N, A * 4, H, W
        """
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()
        
        #########cofs##################
        det_cofs = det_cofs.view(N, 32*4, H, W).permute(0, 2, 3, 1)
        det_cofs = det_cofs.reshape(N, -1, 32*4)

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        box_cls = box_cls * centerness[:, :, None]
        results = []

        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()

            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1] + 1
            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]
            
            #########cofs############
            per_det_cofs = det_cofs[i]
            per_det_cofs = per_det_cofs[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]
                ########cofs############
                per_det_cofs = per_det_cofs[top_k_indices]
            
            
            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)

            h, w = image_sizes[i]
            # print(h,w)
            boxlist = BoxList(detections, (int(w), int(h)), mode="xyxy")
            boxlist.add_field("labels", per_class)
            boxlist.add_field("scores", torch.sqrt(per_box_cls))
            boxlist.add_field("cofs", per_det_cofs)
            boxlist = boxlist.clip_to_image(remove_empty=False)
            results.append(boxlist)

        return results

    def forward(self, locations, box_cls, box_regression, centerness, det_cofs, feat_masks, image_sizes, img_metas=None):
        """
        Arguments:
            anchors: list[list[BoxList]]
            box_cls: list[tensor]
            box_regression: list[tensor]
            image_sizes: list[(h, w)]
        Returns:
            boxlists (list[BoxList]): the post-processed anchors, after
                applying box decoding and NMS
        """

        sampled_boxes = []
        for _, (l, o, b, c, d) in enumerate(zip(locations, box_cls, box_regression, centerness, det_cofs)):
            sampled_boxes.append(self.forward_for_single_feature_map(l, o, b, c, d, image_sizes))


        boxlists = list(zip(*sampled_boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        if not self.bbox_aug_enabled:
            boxlists = self.select_over_all_levels(boxlists, feat_masks, image_sizes, img_metas)

        return boxlists

    # TODO very similar to filter_results from PostProcessor
    # but filter_results is per image
    # TODO Yang: solve this issue in the future. No good solution
    # right now.
    def select_over_all_levels(self, boxlists, feat_masks, image_sizes, img_metas):
        num_images = len(boxlists)
        results = []
        

        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]

            det_cofs = result.get_field("cofs")
            det_bboxes = result.convert("xyxy").bbox


            #########mask####################
            ori_shape = (img_metas[i][1],img_metas[i][0])
            scale_factor = np.minimum(image_sizes[i][0]/img_metas[i][1],image_sizes[i][1]/img_metas[i][0])
            if len(result) > 0:
                scale = 2
                #####SipMask########################
                img_mask1 = feat_masks[i].permute(1, 2, 0)
                pos_masks00 = torch.sigmoid(img_mask1 @ det_cofs[:, 0:32].t())
                pos_masks01 = torch.sigmoid(img_mask1 @ det_cofs[:, 32:64].t())
                pos_masks10 = torch.sigmoid(img_mask1 @ det_cofs[:, 64:96].t())
                pos_masks11 = torch.sigmoid(img_mask1 @ det_cofs[:, 96:128].t())
                pos_masks = self.crop_cuda(torch.stack([pos_masks00,pos_masks01,pos_masks10,pos_masks11],dim=0), det_bboxes/ scale)

                pos_masks = pos_masks.permute(2, 0, 1)
                masks = F.interpolate(pos_masks.unsqueeze(0), scale_factor=scale/scale_factor, mode='bilinear', align_corners=False).squeeze(0)
                masks.gt_(0.4)

                shape = np.minimum(masks.shape[1:3], ori_shape)
                im_masks = torch.zeros((masks.shape[0],ori_shape[0],ori_shape[1]), dtype=torch.uint8, device=masks.device)
                im_masks[:, :shape[0], :shape[1]] = masks[:, :shape[0], :shape[1]]
                im_masks = im_masks[:, None]
            else:
                im_masks = det_bboxes.new_empty((0, 1, ori_shape[0], ori_shape[1]))

            result.add_field("mask", im_masks)
            results.append(result)
        return results

    def select_over_all_levels0(self, boxlists):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            # multiclass nms
            result = boxlist_ml_nms(boxlists[i], self.nms_thresh)
            number_of_detections = len(result)

            # Limit to max_per_image detections **over all classes**
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                cls_scores = result.get_field("scores")
                image_thresh, _ = torch.kthvalue(
                    cls_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            results.append(result)
        return results

def make_sipmask_postprocessor(config):
    pre_nms_thresh = config.MODEL.SIPMASK.INFERENCE_TH
    pre_nms_top_n = config.MODEL.SIPMASK.PRE_NMS_TOP_N
    nms_thresh = config.MODEL.SIPMASK.NMS_TH
    fpn_post_nms_top_n = config.TEST.DETECTIONS_PER_IMG
    bbox_aug_enabled = config.TEST.BBOX_AUG.ENABLED

    box_selector = SipMaskPostProcessor(
        pre_nms_thresh=pre_nms_thresh,
        pre_nms_top_n=pre_nms_top_n,
        nms_thresh=nms_thresh,
        fpn_post_nms_top_n=fpn_post_nms_top_n,
        min_size=0,
        num_classes=config.MODEL.SIPMASK.NUM_CLASSES,
        bbox_aug_enabled=bbox_aug_enabled
    )

    return box_selector
