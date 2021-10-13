import torch
import torch.nn as nn
import torch.nn.functional as func


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[-1]
        # log_preds = F.log_softmax(output, dim=-1)
        log_preds = output
        target = target.to(torch.float)
        if self.reduction == 'sum':
            loss = -log_preds.sum().to(torch.float)
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean().to(torch.float)
        # loss_ = loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction,
        #                                                     ignore_index=self.ignore_index)
        loss_ = loss * self.eps / c + (1 - self.eps) * func.binary_cross_entropy(log_preds, target,
                                                                                 reduction=self.reduction)
        return loss_


class LabelSmoothing(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100, threshold=0.5):
        super(LabelSmoothing, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.threshold = threshold

    def forward(self, pred, target):
        bs, ml, n = pred.size()
        log_pred = func.logsigmoid(pred)
        log_pred = torch.where(log_pred > self.threshold, 1, 0)

        log_preds = log_pred.view(-1, n)
        target = target.view(-1, n).to(torch.float)
        if self.reduction == 'sum':
            loss = -log_preds.sum().to(torch.float)
        else:
            loss = -log_preds.sum(dim=-1).to(torch.float)
            if self.reduction == 'mean':
                loss = loss.mean().to(torch.float)
        loss_ = loss * self.eps / n + (1-self.eps) * func.nll_loss(log_preds, target, reduction=self.reduction,
                                                                   ignore_index=self.ignore_index)
        return loss_
