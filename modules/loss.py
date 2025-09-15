import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from torchvision.ops import box_iou, generalized_box_iou

class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        super().__init__()
        self.cost_class, self.cost_bbox, self.cost_giou = cost_class, cost_bbox, cost_giou

    @torch.no_grad()
    def forward(self, outputs, targets):
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)
        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])
        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(self.box_cxcywh_to_xyxy(out_bbox), self.box_cxcywh_to_xyxy(tgt_bbox))
        C = self.cost_bbox*cost_bbox + self.cost_class*cost_class + self.cost_giou*cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(-1)
        return torch.stack([(x_c-0.5*w), (y_c-0.5*h), (x_c+0.5*w), (y_c+0.5*h)], dim=-1)

class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, eos_coef):
        super().__init__()
        self.num_classes, self.matcher, self.weight_dict = num_classes, matcher, weight_dict
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(outputs["pred_logits"].shape[:2], self.num_classes, dtype=torch.int64, device=outputs["pred_logits"].device)
        target_classes[idx] = target_classes_o
        loss_ce = nn.functional.cross_entropy(outputs['pred_logits'].transpose(1, 2), target_classes, self.empty_weight)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = nn.functional.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_giou = 1 - torch.diag(generalized_box_iou(self.matcher.box_cxcywh_to_xyxy(src_boxes), self.matcher.box_cxcywh_to_xyxy(target_boxes)))
        losses = {'loss_ce': loss_ce, 'loss_bbox': loss_bbox.sum()/len(target_boxes), 'loss_giou': loss_giou.sum()/len(target_boxes)}
        return losses

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx