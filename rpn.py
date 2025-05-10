
import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG16_Weights
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_iou(boxes1,boxes2):
    r"""
    :param boxes1: (N x 4)
    :param boxes2: (M x 4)
    :return: IOU matrix of shape (N x M)
    """
    # Area of boxes (x2-x1)*(y2-y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Get top left x1,y1
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0]) #(N x M)
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1]) #(N x M)
    
    # Get bottom right x2,y2
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])
    
    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(min=0)
    union = area1[:, None] + area2 - intersection_area
    return intersection_area / union # (N x M)

def apply_regression_pred_to_anchors_or_proposals(
    box_transform_pred, anchors_or_proposals
):
    
    r"""
    :param box_transform_pred: (num_anchors_or_proposals, num_classes, 4)
    :param anchors_or_proposals: (num_anchors_or_proposals, 4)
    :return: pred_boxes : (num_anchors_or_proposals, num_classes, 4)
    """
    
    box_transform_pred = box_transform_pred.reshape(
        box_transform_pred.size(0), -1, 4
    )
    
    # Get xs, cy, w, h, from x1, y1, x2, y2
    w = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    h = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * w
    center_y = anchors_or_proposals[:, 1] + 0.5 * h
    
    dx = box_transform_pred[..., 0]
    dy = box_transform_pred[..., 1]
    dw = box_transform_pred[..., 2]
    dh = box_transform_pred[..., 3]
    # dh -> (num_anchors_or_proposals, num_classes)
    
    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]
    # pred_center_x -> (num_anchors_or_proposals, num_classes)
    
    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h
    
    pred_boxes = torch.stack((
        pred_box_x1,
        pred_box_y1,
        pred_box_x2,
        pred_box_y2,
    ), dim=2)
    #pred_boxes -> (num_anchors_or_proposals, num_classes, 4)
    return pred_boxes

def clamp_boxes_to_image_boundary(boxes, image_shape):
    boxes_x1 = boxes[..., 0]
    boxes_y1 = boxes[..., 1]
    boxes_x2 = boxes[..., 2]
    boxes_y2 = boxes[..., 3]
    height, width = image_shape[-2:]
    boxes_x1 = boxes_x1.clamp(min=0, max=width)
    boxes_x2 = boxes_x2.clamp(min=0, max=width)
    boxes_y1 = boxes_y1.clamp(min=0, max=height)
    boxes_y2 = boxes_y2.clamp(min=0, max=height)
    boxes = torch.stack((
        boxes_x1[..., None], 
        boxes_y1[..., None],
        boxes_x2[..., None], 
        boxes_y2[..., None]), dim=-1)
    return boxes

def boxes_to_transformation_targets(ground_truth_boxes, anchors_or_proposals):
    # Get center_x, center_y, w, h from x1, y1, x2, y2 for anchors
    widths = anchors_or_proposals[:, 2] - anchors_or_proposals[:, 0]
    heights = anchors_or_proposals[:, 3] - anchors_or_proposals[:, 1]
    center_x = anchors_or_proposals[:, 0] + 0.5 * widths
    center_y = anchors_or_proposals[:, 1] + 0.5 * heights
    
    # Get center_x, center_y, w, h from x1, y1, x2, y2 for gt boxes
    gt_widths = ground_truth_boxes[:, 2] - ground_truth_boxes[:, 0]
    gt_heights = ground_truth_boxes[:, 3] - ground_truth_boxes[:, 1]
    gt_center_x = ground_truth_boxes[:, 0] + 0.5 * gt_widths
    gt_center_y = ground_truth_boxes[:, 1] + 0.5 * gt_heights
    
    target_dx = (gt_center_x - center_x) / widths
    target_dy = (gt_center_y - center_y) / heights
    target_dw = torch.log(gt_widths / widths)
    target_dh = torch.log(gt_heights / heights)
    
    regression_targets = torch.stack((
        target_dx,
        target_dy,
        target_dw,
        target_dh,
    ), dim=1)
    return regression_targets

def sample_positive_negative(labels, positive_count, total_count):
    positive = torch.where(labels >= 1)[0]
    negative = torch.where(labels == 0)[0]
    # Fix: Changed num_pos to positive_count
    num_pos = min(positive.numel(), positive_count)
    num_neg = total_count - num_pos
    num_neg = min(negative.numel(), num_neg)
    perm_positive_idxs = torch.randperm(positive.numel(), device=positive.device)[:num_pos]
    perm_negative_idxs = torch.randperm(negative.numel(), device=negative.device)[:num_neg]
    pos_idxs = positive[perm_positive_idxs]
    neg_idxs = negative[perm_negative_idxs]
    sampled_pos_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_neg_idx_mask = torch.zeros_like(labels, dtype=torch.bool)
    sampled_pos_idx_mask[pos_idxs] = True
    sampled_neg_idx_mask[neg_idxs] = True
    return sampled_neg_idx_mask, sampled_pos_idx_mask

def transform_boxes_to_original_size(boxes, new_size, original_size):
    ratio = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device) / torch.tensor(s, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratio
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

class RegionPropsalNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super(RegionPropsalNetwork, self).__init__()
        self.scales = [128, 256, 512]
        self.aspect_ratios = [0.5, 1, 2]
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)
        
        # 3*3 conv - Fixed kernel_size typo from 512 to 3
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        
        # 1*1 clasification
        self.cls_layer = nn.Conv2d(in_channels, self.num_anchors * 2, kernel_size=1, stride=1)
        
        # 1*1 regression
        self.bbox_reg_layer = nn.Conv2d(in_channels, self.num_anchors * 4, kernel_size=1, stride=1)
        
    def assign_targets_to_anchors(self, anchors, gt_boxes):
        # Get (gt_boxes, numanchors) IOU matrix
        iou_matrix = get_iou(gt_boxes, anchors)
        
        # For each anchor get best gt box index
        best_match_iou, best_match_gt_index = iou_matrix.max(dim=0)
        
        # This copy will be needed later to add low
        # quality boxes
        best_match_gt_idx_pre_threshold = best_match_gt_index.clone()
        
        below_low_threshold = best_match_iou < 0.3
        between_threshold = (best_match_iou >= 0.3) & (best_match_iou < 0.7)
        best_match_gt_index[below_low_threshold] = -1
        best_match_gt_index[between_threshold] = -2
        
        # Low quality anchor boxes
        best_anchors_iou_for_gt, _ = iou_matrix.max(dim=1)
        gt_pred_pair_with_highest_iou = torch.where(iou_matrix == best_anchors_iou_for_gt[:, None])
        
        # Get all the anchors indexes to update
        pred_inds_to_update = gt_pred_pair_with_highest_iou[1]
        best_match_gt_index[pred_inds_to_update] = best_match_gt_idx_pre_threshold[pred_inds_to_update] 
        
        # best match index is either valid or -1(background) or -2(to ignore)
        matched_gt_boxes = gt_boxes[best_match_gt_index.clamp(min=0)]
        
        # Set all foreground anchor labels as 1
        labels = best_match_gt_index >= 0
        labels = labels.to(dtype=torch.float32)
        
        # Set all background labels as 0
        background_anchors = best_match_gt_index == -1
        labels[background_anchors] = 0.0
        
        # Set all to be ignored anchors labels as -1
        ignored_anchors = best_match_gt_index == -2
        labels[ignored_anchors] = -1.0
        
        # Later for classification we pick labels which hvave >= 0
        return labels, matched_gt_boxes
        
    def filter_proposals(self, proposals, cls_scores, image_shape):
        # Pre NMS Filtering
        cls_scores = cls_scores.reshape(-1)
        # Fixed: Removed duplicate sigmoid call
        cls_scores = cls_scores.sigmoid()
        _, top_n_idx = cls_scores.topk(10000)
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]
        
        # Clamp boxes to image boundary
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)
        
        # NMS Based on objectness
        keep_indices = torch.ops.torchvision.nms(
            proposals, cls_scores, 0.7)
        
        post_nms_keep_indices = keep_indices[
            cls_scores[keep_indices].sort(descending=True)[1]
        ]
        
        # Post NMS topk filtering
        proposals = proposals[post_nms_keep_indices[:2000]]
        cls_scores = cls_scores[post_nms_keep_indices[:2000]]
        return proposals, cls_scores
    
    def generate_anchors(self, image, feat):
        grid_h, grid_w = feat.shape[-2:]
        image_h, image_w = image.shape[-2:]
        
        stride_h = torch.tensor(image_h // grid_h, dtype=torch.int64, device=feat.device)
        stride_w = torch.tensor(image_w // grid_w, dtype=torch.int64, device=feat.device)
        
        scales = torch.as_tensor(self.scales, dtype=feat.dtype, device=feat.device)
        aspect_ratios = torch.as_tensor(self.aspect_ratios, dtype=feat.dtype, device=feat.device)
        
        # The below code ensures h/w = aspect_ratios and h*w = 1
        h_ratios = torch.sqrt(aspect_ratios)
        w_ratios = 1 / h_ratios
        
        ws = (w_ratios[:, None] * scales[None, :]).view(-1)
        hs = (h_ratios[:, None] * scales[None, :]).view(-1)
        
        base_anchors = torch.stack([-ws, -hs, ws, hs], dim=-1) / 2
        base_anchors = base_anchors.round()
        
        # Get the shift in x axis (0, 1, 2, 3, ..., W_feat-1) * stride_w
        shifts_x = torch.arange(0, grid_w, dtype=torch.int32, device=feat.device) * stride_w
        # Get the shift in y axis (0, 1, 2, 3, ..., H_feat-1) * stride_h
        shifts_y = torch.arange(0, grid_h, dtype=torch.int32, device=feat.device) * stride_h
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x, indexing='ij')
        
        # (H_feat, W_feat)
        
        shifts_x = shifts_x.reshape(-1)
        shifts_y = shifts_y.reshape(-1)
        shifts = torch.stack([shifts_x, shifts_y, shifts_x, shifts_y], dim=1)
        
        # shifts -> (H_feat * W_feat, 4)
        
        # base_anchors -> (num_anchors_per_location, 4)
        # shifts -> (H_feat * W_feat, 4)
        anchors = (shifts.view(-1, 1, 4) + base_anchors.view(1, -1, 4))
        # (H_feat * W_feat * num_anchors_per_location, 4)
        
        anchors = anchors.reshape(-1, 4)
        # anchors -> (H_feat * W_feat * num_anchors_per_location, 4)
        return anchors
    
    def forward(self, image, feat, target):
        # Call RPN Layers
        rpn_feat = nn.functional.relu(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)
    
    # Generate anchors
        anchors = self.generate_anchors(image, feat)
    
    # Get dimensions for reshaping
        batch_size = feat.size(0)
        feat_h, feat_w = feat.shape[-2:]
    
    # cls_scores -> (Batch, Numbers of Anchors per location * 2, H_feat, W_feat)
        number_of_anchors_per_location = cls_scores.size(1) // 2  # Divide by 2 for foreground/background
    
    # Reshape cls_scores to get one score per anchor
        cls_scores = cls_scores.view(batch_size, 2, number_of_anchors_per_location, feat_h, feat_w)
        cls_scores = cls_scores.permute(0, 2, 3, 4, 1)  # [batch, anchors_per_loc, h, w, 2]
        cls_scores = cls_scores.reshape(batch_size, -1, 2)  # [batch, num_anchors, 2]
        cls_scores = cls_scores.reshape(-1, 2)[:, 0:1]  # Take foreground score
    
    # Reshape box_transform_pred to get one prediction per anchor
        box_transform_pred = box_transform_pred.view(
            batch_size,
            number_of_anchors_per_location,
            4,
            feat_h,
            feat_w
        )
        box_transform_pred = box_transform_pred.permute(0, 1, 3, 4, 2)  # [batch, anchors_per_loc, h, w, 4]
        box_transform_pred = box_transform_pred.reshape(batch_size, -1, 4)  # [batch, num_anchors, 4]
        box_transform_pred = box_transform_pred.reshape(-1, 4)  # [batch*num_anchors, 4]
    
    # Make sure anchors match the batch size
    # If we have multiple images in the batch, repeat anchors for each image
        if batch_size > 1:
            anchors = anchors.repeat(batch_size, 1)
    
    # Now the number of anchors should match the number of box transformations
        assert anchors.size(0) == box_transform_pred.size(0), f"Anchor count ({anchors.size(0)}) doesn't match box transform count ({box_transform_pred.size(0)})"
    
    # Transform generated anchors according to box_transform_pred
        proposals = apply_regression_pred_to_anchors_or_proposals(
            box_transform_pred.detach().reshape(-1, 1, 4),
            anchors
        )
        proposals = proposals.reshape(proposals.size(0), 4)
        proposals, scores = self.filter_proposals(
            proposals, cls_scores, image.shape
        )
    
        rpn_output = {
            'proposals': proposals,
            'scores': scores,
        }
        
        if not self.training or target is None:
            return rpn_output
        else:
            # in training
            # Assign gt box and label for each anchor
            # FIX: Handle target as a list of dictionaries
            # Get the bboxes from the first item in the target list
            target_bboxes = target[0]['bboxes'] if isinstance(target, list) else target['bboxes']
            
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors,
                target_bboxes
            )
            
            # Based on gt assignment above, get regression targets for anchors
            # matched_gt_boxes_for_anchors -> (num_anchors in image, 4)
            # anchors -> (num_anchors in image, 4)
            regression_targets = boxes_to_transformation_targets(
                matched_gt_boxes_for_anchors, anchors
            )
            
            # Sample positive and negative anchors for training
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels_for_anchors,
                positive_count=128,
                total_count=256
            )
            sampled_idxs = torch.where(sampled_neg_idx_mask | sampled_pos_idx_mask)[0]
            
            # Fixed: Removed the dictionary wrapping for localization_loss
            localization_loss = torch.nn.functional.smooth_l1_loss(
                box_transform_pred[sampled_pos_idx_mask],
                regression_targets[sampled_pos_idx_mask],
                beta=1 / 9,
                reduction='sum'
            ) / (sampled_idxs.numel())
            
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_scores[sampled_idxs].flatten(),
                labels_for_anchors[sampled_idxs].float(),
            )
            
            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = localization_loss
            return rpn_output
                
        
class ROIHead(nn.Module):
    def __init__(self, num_classes=21, in_channels=512):
        # Fixed: Changed __int__ to __init__
        super().__init__()
        self.num_classes = num_classes
        self.pool_size = 7
        self.fc_inner_dim = 1024
        
        self.fc6 = nn.Linear(in_channels * self.pool_size * self.pool_size, self.fc_inner_dim)
        self.fc7 = nn.Linear(self.fc_inner_dim, self.fc_inner_dim)
        self.cls_layer = nn.Linear(self.fc_inner_dim, self.num_classes)
        self.bbox_reg_layer = nn.Linear(self.fc_inner_dim, self.num_classes * 4)
        
    def assign_target_to_proposals(self, proposals, gt_boxes, gt_labels):
        iou_matrix = get_iou(gt_boxes, proposals)
        best_match_iou, best_match_gt_idx = iou_matrix.max(dim=0)
        below_low_threshold = best_match_iou < 0.5
        
        best_match_gt_idx[below_low_threshold] = -1
        matched_gt_boxes_for_proposals = gt_boxes[best_match_gt_idx.clamp(min=0)]
        
        labels = gt_labels[best_match_gt_idx.clamp(min=0)]
        labels = labels.to(dtype=torch.int64)
        
        background_proposals = best_match_gt_idx == -1
        labels[background_proposals] = 0
        return labels, matched_gt_boxes_for_proposals
        
    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        # remove low scoring boxes
        keep = torch.where(pred_scores > 0.05)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        # remove small boxes
        min_size = 1
        ws, hs = pred_boxes[:, 2] - pred_boxes[:, 0], pred_boxes[:, 3] - pred_boxes[:, 1]
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        
        # class wise nms
        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(pred_labels == class_id)[0]
            curr_keep_indices = torch.ops.torchvision.nms(
                pred_boxes[curr_indices],
                pred_scores[curr_indices],
                0.5,
            )
            keep_mask[curr_indices[curr_keep_indices]] = True
        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[pred_scores[keep_indices].sort(
            descending=True
            )[1]]
        keep = post_nms_keep_indices[:100]
        pred_boxes, pred_scores, pred_labels = pred_boxes[keep], pred_scores[keep], pred_labels[keep]
        return pred_boxes, pred_scores, pred_labels   
        
    def forward(self, image, proposals, feat, target):
        if self.training and target is not None:
            # FIX: Handle target as a list of dictionaries
            # Get the bboxes and labels from the first item in the target list
            gt_boxes = target[0]['bboxes'] if isinstance(target, list) else target['bboxes']
            gt_labels = target[0]['labels'] if isinstance(target, list) else target['labels']
            
            # assign labels and gt boxes for proposals
            labels, matched_gt_boxes_for_proposals = self.assign_target_to_proposals(
                proposals, gt_boxes, gt_labels
            )
            
            sampled_neg_idx_mask, sampled_pos_idx_mask = sample_positive_negative(
                labels, positive_count=32, total_count=128
            )
            sampled_idxs = torch.where(sampled_neg_idx_mask | sampled_pos_idx_mask)[0]
           
            proposals = proposals[sampled_idxs]
            labels = labels[sampled_idxs]
            matched_gt_boxes_for_proposals = matched_gt_boxes_for_proposals[sampled_idxs]
            regression_targets = boxes_to_transformation_targets(
                matched_gt_boxes_for_proposals, proposals
            )
            # regression_targets -> (sampled_training_proposal, 4)
            
        # ROI pooling part
        # spatial scale for roi pooling
        # For vgg16 this would be 1/16
        spatial_scale = 0.0625
        
        # Added missing ROI pooling implementation
        proposal_roi_pool_feats = torchvision.ops.roi_pool(
            feat, 
            [proposals.detach()], 
            output_size=(self.pool_size, self.pool_size),
            spatial_scale=spatial_scale
        )
                
        proposal_roi_pool_feats = proposal_roi_pool_feats.flatten(start_dim=1)
        box_fc_6 = torch.nn.functional.relu(self.fc6(proposal_roi_pool_feats))
        box_fc_7 = torch.nn.functional.relu(self.fc7(box_fc_6)) 
        cls_scores = self.cls_layer(box_fc_7)
        box_transform_pred = self.bbox_reg_layer(box_fc_7)
            
        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(
                num_boxes, num_classes, 4
            )
        frcnn_output = {}
        if self.training and target is not None:
                classification_loss = torch.nn.functional.cross_entropy(
                    cls_scores,
                    labels
                )
                
                # Compute localization only for non-background
                fg_proposal_idxs = torch.where(labels > 0)[0]
                # Get class labels for them
                fg_class_labels = labels[fg_proposal_idxs]
                localization_loss = torch.nn.functional.smooth_l1_loss(
                    box_transform_pred[fg_proposal_idxs, fg_class_labels],
                    regression_targets[fg_proposal_idxs],
                    beta=1 / 9,
                    reduction='sum'
                )
                localization_loss = localization_loss / labels.numel()
                frcnn_output['frcnn_classification_loss'] = classification_loss
                frcnn_output['frcnn_localization_loss'] = localization_loss
                return frcnn_output
        else:
                # Apply transformation prediction to proposals
                pred_boxes = apply_regression_pred_to_anchors_or_proposals(
                    box_transform_pred, proposals
                )
                pred_scores = torch.nn.functional.softmax(cls_scores, dim=1)
                
                # Clamp boxes to image boundary
                pred_boxes = clamp_boxes_to_image_boundary(pred_boxes, image.shape)
                
                # Create labels for each prediction
                pred_labels = torch.arange(num_classes, device=cls_scores.device)
                pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)
                
                # remove background class prediction
                pred_boxes = pred_boxes[:, 1:]
                pred_scores = pred_scores[:, 1:]
                pred_labels = pred_labels[:, 1:]
                # pred_boxes -> (num_proposals, num_classes-1, 4)
                
                # Batch everything by making every class prediction a separate
                # instance
                
                pred_boxes = pred_boxes.reshape(-1, 4)
                pred_scores = pred_scores.reshape(-1)
                pred_labels = pred_labels.reshape(-1)
                
                pred_boxes, pred_scores, pred_labels = self.filter_predictions(
                    pred_boxes, pred_labels, pred_scores
                )
                frcnn_output['boxes'] = pred_boxes
                frcnn_output['scores'] = pred_scores
                frcnn_output['labels'] = pred_labels
                return frcnn_output
                
class FasterRCNN(nn.Module):
    def __init__(self, num_classes=21):
        super(FasterRCNN, self).__init__()
        vgg16 = torchvision.models.vgg16(weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1)  # Updated weights parameter
        self.backbone = vgg16.features[:-1]
        self.rpn = RegionPropsalNetwork(in_channels=512)
        self.roi_head = ROIHead(num_classes=num_classes) 
            
        for layer in self.backbone[:10]:
            for p in layer.parameters():
                p.requires_grad = False
        self.image_mean = [0.485, 0.456, 0.406]
        self.image_std = [0.229, 0.224, 0.225]
        self.min_size = 600
        self.max_size = 1000
        
    def normalize_resize_image_and_boxes(self, image, bboxes):
    # Normalize
        mean = torch.as_tensor(self.image_mean,
                           dtype=image.dtype,
                           device=image.device)
        std = torch.as_tensor(self.image_std,
                         dtype=image.dtype,
                         device=image.device)
        image = (image - mean[:, None, None]) / std[:, None, None]
    ##########
    
    # Resize such that lower dim is scaled to 600
    # but larger dim not more than 1000
        h,w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        min_size = torch.min(im_shape).to(dtype=torch.float32)
        max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(
            float(self.min_size) / min_size,
            float(self.max_size) / max_size
        )
        scale_factor = scale.item()
    # Resize image based on scale
        image = torch.nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale_factor,
            mode='bilinear',
            recompute_scale_factor=True,
            align_corners=False,
        )
    
    # Resize bboxes
        if bboxes is not None:
        # Check the shape of bboxes to determine how to process it
            if len(bboxes.shape) == 3:  # [batch, num_boxes, coords]
            # Check if we have 4 coordinates per box
                if bboxes.shape[2] == 4:  # Standard [x1, y1, x2, y2] format
                    ratios = [
                        torch.tensor(s, dtype=torch.float32, device=bboxes.device) / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                        for s, s_orig in zip(image.shape[-2:], (h, w))
                    ]
                    ratio_height, ratio_width = ratios
                
                # Process each box in the batch
                    resized_bboxes = []
                    for batch_boxes in bboxes:
                        xmin, ymin, xmax, ymax = batch_boxes.unbind(1)
                        xmin = xmin * ratio_width
                        xmax = xmax * ratio_width
                        ymin = ymin * ratio_height
                        ymax = ymax * ratio_height
                        resized_batch_boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
                        resized_bboxes.append(resized_batch_boxes)
                
                    bboxes = torch.stack(resized_bboxes)
                else:  # Non-standard format, print debug info
                    print(f"Warning: Unexpected bounding box format. Shape: {bboxes.shape}")
                # Try to handle [batch, num_boxes, 2] format (might be [center_x, center_y, width, height])
                    if bboxes.shape[2] == 2:
                    # This is likely a different format, let's print it for debugging
                        print("Bounding box values (first few):", bboxes[0, :2])
                    # For now, return the original bboxes
                        return image, bboxes
            elif len(bboxes.shape) == 2:  # [num_boxes, coords]
            # Similar handling for 2D tensor
                ratios = [
                    torch.tensor(s, dtype=torch.float32, device=bboxes.device) / torch.tensor(s_orig, dtype=torch.float32, device=bboxes.device)
                    for s, s_orig in zip(image.shape[-2:], (h, w))
                ]
                ratio_height, ratio_width = ratios
            
            # Check if we have 4 coordinates per box
                if bboxes.shape[1] == 4:  # Standard [x1, y1, x2, y2] format
                    xmin, ymin, xmax, ymax = bboxes.unbind(1)
                    xmin = xmin * ratio_width
                    xmax = xmax * ratio_width
                    ymin = ymin * ratio_height
                    ymax = ymax * ratio_height
                    bboxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
                else:  # Non-standard format, print debug info
                    print(f"Warning: Unexpected bounding box format. Shape: {bboxes.shape}")
                # Try to handle [num_boxes, 2] format
                    if bboxes.shape[1] == 2:
                        print("Bounding box values (first few):", bboxes[:2])
                    # For now, return the original bboxes
                        return image, bboxes
        
            return image, bboxes
        return image, None
          
    def forward(self, image, target=None):
        old_shape = image.shape[-2:]
        if self.training:
            # FIX: Handle target as a list of dictionaries
            if target is not None and isinstance(target, list):
                # Normalize and resize boxes for each image in the batch
                image_list = []
                target_list = []
                
                for i, (img, tgt) in enumerate(zip(image, target)):
                    img_resized, bboxes_resized = self.normalize_resize_image_and_boxes(img.unsqueeze(0), tgt['bboxes'].unsqueeze(0))
                    image_list.append(img_resized.squeeze(0))
                    
                    # Update the target with resized bboxes
                    tgt_resized = tgt.copy()
                    tgt_resized['bboxes'] = bboxes_resized.squeeze(0)
                    target_list.append(tgt_resized)
                
                # Stack images back into a batch
                image = torch.stack(image_list)
                target = target_list
            else:
                # Handle single target case (for backward compatibility)
                image, bboxes = self.normalize_resize_image_and_boxes(image, target['bboxes'])
                if target is not None:
                    target['bboxes'] = bboxes
        else:
            image, _ = self.normalize_resize_image_and_boxes(image, None)
            
        # Call backbone
        feat = self.backbone(image)
        
        # Call RPN and get proposals
        rpn_output = self.rpn(image, feat, target)
        proposals = rpn_output['proposals']
        
        # Call ROI head and convert proposals to boxes
        frcnn_output = self.roi_head(image, proposals, feat, target)
        
        if not self.training:
            # transform boxes to original image dimension
            frcnn_output['boxes'] = transform_boxes_to_original_size(
                frcnn_output['boxes'], image.shape[-2:], old_shape
            )
        return rpn_output, frcnn_output

# Example usage
def test_model():
    # Create a dummy image and target
    image = torch.randn(1, 3, 800, 800)
    target = {
        'bboxes': torch.tensor([[[100, 100, 200, 200], [300, 300, 400, 400]]], dtype=torch.float32),
        'labels': torch.tensor([[1, 2]], dtype=torch.int64)
    }
    
    # Initialize model
    model = FasterRCNN(num_classes=21)
    model.eval()  # Set to evaluation mode
    
    # Forward pass
    with torch.no_grad():
        rpn_output, frcnn_output = model(image)
    
    print("RPN output keys:", rpn_output.keys())
    print("Number of proposals:", rpn_output['proposals'].shape)
    
    if 'boxes' in frcnn_output:
        print("FRCNN output - detected boxes:", frcnn_output['boxes'].shape)
        print("FRCNN output - scores:", frcnn_output['scores'].shape)
        print("FRCNN output - labels:", frcnn_output['labels'].shape)
    
    return "Model test completed successfully"

# Uncomment to test
# test_model()