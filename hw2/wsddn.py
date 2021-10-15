import torch
import torchvision
import torch.utils.data as data
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

import numpy as np
import torchvision.models as models

from torchvision.ops import roi_pool, roi_align


class WSDDN(nn.Module):
    n_classes = 20
    classes = np.asarray([
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ])

    def __init__(self, classes=None):
        super(WSDDN, self).__init__()

        if classes is not None:
            self.classes = classes
            self.n_classes = len(classes)
            print(classes)

        # TODO: Define the WSDDN model
        self.features   = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(64, 192, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), dilation=(1, 1), ceil_mode=False),
            nn.Conv2d(192, 384, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self.score_fc   = nn.Linear(4096, self.n_classes)
        self.bbox_fc    = nn.Linear(4096, self.n_classes)

        # loss
        self.cross_entropy = None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self,
                image,
                rois=None,
                gt_vec=None,
                ):

        # TODO: Use image and rois as input
        # compute cls_prob which are N_roi X 20 scores
        feature_map = self.features(image)
        N, C, H, W = feature_map.shape
        rois_ = torch.zeros(rois.shape[1:]).cuda()
        rois_[:, 0] = rois[0, :, 0]
        rois_[:, 1] = rois[0, :, 1]
        rois_[:, 2] = rois[0, :, 2]
        rois_[:, 3] = rois[0, :, 3]
        rois_ = torch.cat((torch.zeros((rois_.shape[0], 1)).cuda(), rois_), 1)

        roi_out = roi_pool(feature_map, rois_, (6, 6), spatial_scale=31/512).view(-1, 256 * 6 * 6)
        classifier_out = self.classifier(roi_out)

        score_out = self.score_fc(classifier_out)
        score_softmax = torch.softmax(score_out, dim=1)
        bbox_out = self.bbox_fc(classifier_out)
        bbox_softmax = torch.softmax(bbox_out, dim=0)
        cls_prob = score_softmax * bbox_softmax

        if self.training:
            label_vec = gt_vec.view(-1, self.n_classes)
            self.cross_entropy = self.build_loss(cls_prob, label_vec)
        
        return cls_prob

    def build_loss(self, cls_prob, label_vec):
        """Computes the loss

        :cls_prob: N_roix20 output scores
        :label_vec: 1x20 one hot label vector
        :returns: loss

        """
        # TODO: Compute the appropriate loss using the cls_prob that is the
        # output of forward()
        # Checkout forward() to see how it is called
        cls_prob = torch.sum(cls_prob, dim=0, keepdim=True)
        return nn.BCELoss(reduction="sum")(torch.clamp(cls_prob, min=0.0, max=1.0), label_vec)
