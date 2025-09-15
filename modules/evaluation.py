import torch
from torchvision.ops import box_iou

@torch.no_grad()
def evaluate(model, criterion, dataloader, device, conf_threshold=0.5):
    """
    Evaluation loop to compute validation loss, precision, and recall.
    """
    model.eval()
    total_loss = 0.0
    
    # Statistics for calculating precision and recall
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        outputs = model(images)
        
        # --- 1. Calculate Loss ---
        losses_dict = criterion(outputs, targets)
        weighted_loss = sum(losses_dict[k] * criterion.weight_dict[k.replace('loss_', '')] for k in losses_dict.keys())
        total_loss += weighted_loss.item()
        
        # --- 2. Calculate Precision and Recall ---
        probs = outputs['pred_logits'].softmax(-1)
        scores, labels = probs.max(-1)
        
        for i in range(len(targets)): # Iterate over each image in the batch
            # Filter predictions by confidence threshold
            keep = scores[i] > conf_threshold
            pred_boxes = outputs['pred_boxes'][i][keep]
            pred_labels = labels[i][keep]
            
            gt_boxes = targets[i]['boxes']
            gt_labels = targets[i]['labels']
            
            if len(pred_boxes) == 0:
                # If no predictions, all ground truths are missed
                false_negatives += len(gt_boxes)
                continue

            if len(gt_boxes) == 0:
                # If no ground truths, all predictions are false positives
                false_positives += len(pred_boxes)
                continue

            # Calculate IoU between all predictions and all ground truths
            iou = box_iou(box_cxcywh_to_xyxy(pred_boxes), box_cxcywh_to_xyxy(gt_boxes))
            
            # Find the best GT match for each prediction
            # `max(dim=1)` gives (max_iou_values, max_iou_indices)
            max_iou, best_gt_indices = iou.max(dim=1)
            
            # Match class labels
            matching_labels = (pred_labels == gt_labels[best_gt_indices])
            
            # A prediction is a True Positive if IoU > 0.5 and classes match
            is_tp = (max_iou > 0.5) & matching_labels
            
            # Update counts
            tp = torch.sum(is_tp)
            fp = len(pred_boxes) - tp
            
            true_positives += tp.item()
            false_positives += fp.item()
            # FNs are GTs that were not matched by any TP prediction
            # A simple way to estimate is total GTs - TPs found for them
            false_negatives += len(gt_boxes) - tp.item()

    # Calculate final metrics
    precision = true_positives / (true_positives + false_positives + 1e-6)
    recall = true_positives / (true_positives + false_negatives + 1e-6)
    
    return total_loss / len(dataloader), precision, recall

def box_cxcywh_to_xyxy(x):
    """Converts boxes from [center_x, center_y, width, height] to [x1, y1, x2, y2]."""
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)