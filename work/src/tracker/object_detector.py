import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, nms_thresh=0.5):
        backbone = resnet_fpn_backbone('resnet50', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        self.roi_heads.nms_thresh = nms_thresh

    def detect(self, img, conf_thresh=0.5):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]
        
        boxes = detections['boxes'].detach().cpu()
        scores = detections['scores'].detach().cpu()
        
        mask = scores > conf_thresh
        filtered_boxes = boxes[mask]
        filtered_scores = scores[mask]

        return filtered_boxes, filtered_scores