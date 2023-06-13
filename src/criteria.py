import torch
import torch.nn as nn


class ContrastiveLoss(torch.nn.Module):

    def __init__(self, margin=.5, **kwargs):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        # self.metric = metric
        self.distance = torch.nn.PairwiseDistance(p=2)

    def forward(self, out0, out1, label):
        gt = label.float()
        D = self.distance(out0, out1).float().squeeze()
        loss = (1-gt) * 0.5 * torch.pow(D, 2) + (gt) * 0.5 * torch.pow(torch.clamp(self.margin - D, min=0.0), 2)
        return loss


class ContrastivePredictiveLoss(torch.nn.Module):

    def __init__(self, margin=.5, alpha=1, **kwargs):
        super(ContrastivePredictiveLoss, self).__init__()
        self.cl_loss = ContrastiveLoss(margin)
        self.pr_loss = nn.CrossEntropyLoss()
        self.alpha = alpha  # Weighs predictive loss compared to contrastive loss

    def forward(self, out_c0, out_c1, out_p, label_c, label_p):
        loss = self.cl_loss(out_c0, out_c1, label_c) + self.alpha * self.pr_loss(out_p, label_p)
        return loss
