import torch
import torch.nn as nn
import torchvision
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
    num_pos = min(positive.numel(), num_pos)
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

class RegionPropsalNetwork(nn.Module):
    def __init__(self, in_channels=512):
        super(RegionPropsalNetwork, self).__init__()
        self.scales = [128, 256, 512]
        self.aspect_ratios = [0.5, 1, 2]
        self.num_anchors = len(self.scales) * len(self.aspect_ratios)
        
        # 3*3 conv
        self.rpn_conv = nn.Conv2d(in_channels, in_channels, 512, kernel_size=3,stride=1, padding=1)
        
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
        cls_scores = cls_scores.sigmoid(cls_scores)
        _, top_n_idx = cls_scores.topk(10000)
        cls_scores = cls_scores[top_n_idx]
        proposals = proposals[top_n_idx]
        
        # Clamp boxes to image boundary
        proposals = clamp_boxes_to_image_boundary(proposals, image_shape)
        
        # NMS Based on objectness
        keep_mask = torch.zeros_like(cls_scores, dtype=torch.bool)
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
        rpn_feat = nn.ReLU()(self.rpn_conv(feat))
        cls_scores = self.cls_layer(rpn_feat)
        box_transform_pred = self.bbox_reg_layer(rpn_feat)
        
        # Genarate anchors
        anchors = self.generate_anchors(image, feat)
        
        # cls_scores -> (Batch, Numbers of Anchors per location, H_feat, W_feat)
        number_of_anchors_per_location = cls_scores.size(1)
        cls_scores = cls_scores.permute(0, 2, 3, 1)
        cls_scores = cls_scores.reshape(-1, 1)
        # cls_scores -> (Batch * H_feat * W_feat * num_anchors_per_location, 1)
        
        # box_transform_pred -> (Batch, Numbers of Anchors per location * 4, H_feat, W_feat)
        box_transform_pred = box_transform_pred.view(
            box_transform_pred.size(0),
            number_of_anchors_per_location,
            4,
            rpn_feat.shape[-2],
            rpn_feat.shape[-2],
        )
        
        box_transform_pred = box_transform_pred.permute(0, 3, 4, 1, 2)
        box_transform_pred = box_transform_pred.reshape(-1, 4)
        # box_transform_pred -> (Batch * H_feat * W_feat * num_anchors_per_location, 4)
        
        # Transform genarated anchors according to box_transform_pred
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
            labels_for_anchors, matched_gt_boxes_for_anchors = self.assign_targets_to_anchors(
                anchors,
                target['bboxes'][0]
            )
            
            # Based on gt assifnment above, get regression targets for anchors
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
            localization_loss = {
                torch.nn.functional.smooth_l1_loss(
                    box_transform_pred[sampled_pos_idx_mask],
                    regression_targets[sampled_pos_idx_mask],
                    beta=1 / 9,
                    reduction='sum'
                ) / (sampled_idxs.numel())
            }
            
            cls_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                cls_scores[sampled_idxs].flatten(),
                labels_for_anchors[sampled_idxs].float(),
            )
            
            rpn_output['rpn_classification_loss'] = cls_loss
            rpn_output['rpn_localization_loss'] = localization_loss
            return rpn_output
                
        
class ROIHead(nn.Module):
    def __init__(self, num_classes=21, in_channels=512):
        super(ROIHead, self).__int__()
        self.num_classes = num_classes
        self.pool_size=7
        self.fc_inner_dim=1024
        
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
        
        
    def forward(self, image, proposals, feat, target):
        if self.training and target is not None:
            gt_boxes = target['bboxes'][0]
            gt_labels = target['labels'][0]
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
            spatial_scale = 0.0625\
                
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
            if self.training and targets is not None:
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
                
                # Batch everything by making every class prediction a separete
                # insrtance
                
                pred_boxes = pred_boxes.reshape(-1, 4)
                pred_scores = pred_scores.reshape(-1)
                pred_labels = pred_labels.reshape(-1)
                
                